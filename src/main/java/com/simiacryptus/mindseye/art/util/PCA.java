/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.art.util;

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.BandReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer;
import com.simiacryptus.mindseye.layers.cudnn.SquareActivationLayer;
import com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.lang.RecycleBin;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

public class PCA {
  private boolean recenter;
  private boolean rescale;
  private double eigenvaluePower;

  public PCA(boolean recenter, boolean rescale, double eigenvaluePower) {
    this.setRecenter(recenter);
    this.setRescale(rescale);
    this.setEigenvaluePower(eigenvaluePower);
  }

  public PCA() {
    this(true, false, 0.0);
  }

  public double getEigenvaluePower() {
    return eigenvaluePower;
  }

  @Nonnull
  public PCA setEigenvaluePower(double eigenvaluePower) {
    this.eigenvaluePower = eigenvaluePower;
    return this;
  }

  public boolean isRecenter() {
    return recenter;
  }

  @Nonnull
  public PCA setRecenter(boolean recenter) {
    this.recenter = recenter;
    return this;
  }

  public boolean isRescale() {
    return rescale;
  }

  @Nonnull
  public PCA setRescale(boolean rescale) {
    this.rescale = rescale;
    return this;
  }

  public static double[] bandCovariance(@Nonnull final RefStream<double[]> pixelStream, final int pixels, final double[] mean,
                                        final double[] rms) {
    return RefArrays.stream(RefUtil.get(pixelStream.map(pixel -> {
      double[] crossproduct = RecycleBin.DOUBLES.obtain(pixel.length * pixel.length);
      int k = 0;
      for (int j = 0; j < pixel.length; j++) {
        for (int i = 0; i < pixel.length; i++) {
          crossproduct[k++] = ((pixel[i] - mean[i]) / (rms[i] == 0 ? 1 : rms[i]))
              * ((pixel[j] - mean[j]) / (rms[j] == 0 ? 1 : rms[j]));
        }
      }
      RecycleBin.DOUBLES.recycle(pixel, pixel.length);
      return crossproduct;
    }).reduce((a, b) -> {
      for (int i = 0; i < a.length; i++) {
        a[i] += b[i];
      }
      RecycleBin.DOUBLES.recycle(b, b.length);
      return a;
    }))).map(x -> x / pixels).toArray();
  }

  public static int countPixels(@Nonnull final Tensor featureImage) {
    int[] dimensions = featureImage.getDimensions();
    int width = dimensions[0];
    int height = dimensions[1];
    return width * height;
  }

  public static RefList<Tensor> pca(@Nonnull final double[] bandCovariance, final double eigenPower) {
    @Nonnull final EigenDecomposition decomposition = new EigenDecomposition(toMatrix(bandCovariance));
    return RefIntStream.range(0, (int) Math.sqrt(bandCovariance.length)).mapToObj(vectorIndex -> {
      double[] data = decomposition.getEigenvector(vectorIndex).toArray();
      Tensor tensor = new Tensor(data, 1, 1, data.length).unit();
      tensor.scaleInPlace(Math.pow(decomposition.getRealEigenvalue(vectorIndex), eigenPower));
      return tensor.addRef();
    }).collect(RefCollectors.toList());
  }

  @Nonnull
  private static Array2DRowRealMatrix toMatrix(@Nonnull final double[] covariance) {
    final int bands = (int) Math.sqrt(covariance.length);
    Array2DRowRealMatrix matrix = new Array2DRowRealMatrix(bands, bands);
    int k = 0;
    for (int x = 0; x < bands; x++) {
      for (int y = 0; y < bands; y++) {
        matrix.setEntry(x, y, covariance[k++]);
      }
    }
    return matrix;
  }

  public RefList<Tensor> channelPCA(@Nonnull Tensor image) {
    Tensor meanTensor = getChannelMeans(image);
    Tensor scaled = getChannelRms(image, image.getDimensions()[2], meanTensor);
    assert scaled != null;
    double[] bandCovariance = bandCovariance(image.getPixelStream(), countPixels(image), meanTensor.getData(),
        scaled.getData());
    meanTensor.freeRef();
    scaled.freeRef();
    return pca(bandCovariance, getEigenvaluePower()).stream().collect(RefCollectors.toList());
  }

  @Nullable
  public Tensor getChannelRms(Tensor image, int bands, @Nonnull Tensor meanTensor) {
    if (!isRescale()) {
      return meanTensor.map(x -> 1);
    } else {
      NthPowerActivationLayer nthPowerActivationLayer = new NthPowerActivationLayer();
      nthPowerActivationLayer.setPower(0.5);
      BandReducerLayer bandReducerLayer = new BandReducerLayer();
      bandReducerLayer.setMode(PoolingLayer.PoolingMode.Avg);
      ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(bands);
      imgBandBiasLayer.set(meanTensor.scale(-1));
      PipelineNetwork network = PipelineNetwork.build(1, imgBandBiasLayer.addRef(),
          new SquareActivationLayer(), bandReducerLayer.addRef(),
          nthPowerActivationLayer.addRef());
      try {
        return network.eval(image).getData().get(0).map(x -> x == 0.0 ? 1.0 : x);
      } finally {
        network.freeRef();
      }
    }
  }

  @Nonnull
  public Tensor getChannelMeans(Tensor image) {
    BandReducerLayer bandReducerLayer = new BandReducerLayer();
    bandReducerLayer.setMode(PoolingLayer.PoolingMode.Avg);
    Tensor meanTensor = bandReducerLayer.addRef().eval(image).getData().get(0);
    bandReducerLayer.freeRef();
    if (!isRecenter())
      RefArrays.fill(meanTensor.getData(), 0);
    return meanTensor;
  }

}
