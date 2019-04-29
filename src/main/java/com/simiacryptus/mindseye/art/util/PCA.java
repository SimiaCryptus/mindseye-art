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

import com.simiacryptus.lang.ref.RecycleBin;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.BandReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer;
import com.simiacryptus.mindseye.layers.cudnn.SquareActivationLayer;
import com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

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

  public static double[] bandCovariance(final Stream<double[]> pixelStream, final int pixels, final double[] mean, final double[] rms) {
    return Arrays.stream(pixelStream.map(pixel -> {
      double[] crossproduct = RecycleBin.DOUBLES.obtain(pixel.length * pixel.length);
      int k = 0;
      for (int j = 0; j < pixel.length; j++) {
        for (int i = 0; i < pixel.length; i++) {
          crossproduct[k++] = ((pixel[i] - mean[i]) / (rms[i] == 0 ? 1 : rms[i])) * ((pixel[j] - mean[j]) / (rms[j] == 0 ? 1 : rms[j]));
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
    }).get()).map(x -> x / pixels).toArray();
  }

  public static int countPixels(final Tensor featureImage) {
    int[] dimensions = featureImage.getDimensions();
    int width = dimensions[0];
    int height = dimensions[1];
    return width * height;
  }

  public static List<Tensor> pca(final double[] bandCovariance, final double eigenPower) {
    @Nonnull final EigenDecomposition decomposition = new EigenDecomposition(toMatrix(bandCovariance));
    return IntStream.range(0, (int) Math.sqrt(bandCovariance.length)).mapToObj(vectorIndex -> {
      double[] data = decomposition.getEigenvector(vectorIndex).toArray();
      return new Tensor(data, 1, 1, data.length).unit().scaleInPlace(Math.pow(decomposition.getRealEigenvalue(vectorIndex), eigenPower));
    }).collect(Collectors.toList());
  }

  @Nonnull
  private static Array2DRowRealMatrix toMatrix(final double[] covariance) {
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

  public List<Tensor> channelPCA(Tensor image) {
    Tensor meanTensor = getChannelMeans(image);
    Tensor scaled = getChannelRms(image, image.getDimensions()[2], meanTensor);
    double[] bandCovariance = bandCovariance(image.getPixelStream(), countPixels(image), meanTensor.getData(), scaled.getData());
    meanTensor.freeRef();
    scaled.freeRef();
    return pca(bandCovariance, getEigenvaluePower()).stream().collect(Collectors.toList());
  }

  public Tensor getChannelRms(Tensor image, int bands, Tensor meanTensor) {
    if (!isRescale()) {
      return meanTensor.map(x -> 1);
    } else {
      PipelineNetwork network = PipelineNetwork.wrap(1,
          new ImgBandBiasLayer(bands).set(meanTensor.scale(-1)),
          new SquareActivationLayer(),
          new BandReducerLayer().setMode(PoolingLayer.PoolingMode.Avg),
          new NthPowerActivationLayer().setPower(0.5)
      );
      try {
        return network.eval(image).getDataAndFree().getAndFree(0).mapAndFree(x -> x == 0.0 ? 1.0 : x);
      } finally {
        network.freeRef();
      }
    }
  }

  @NotNull
  public Tensor getChannelMeans(Tensor image) {
    BandReducerLayer bandReducerLayer = new BandReducerLayer();
    Tensor meanTensor = bandReducerLayer.setMode(PoolingLayer.PoolingMode.Avg).eval(image).getDataAndFree().getAndFree(0);
    bandReducerLayer.freeRef();
    if (!isRecenter()) Arrays.fill(meanTensor.getData(), 0);
    return meanTensor;
  }

  public boolean isRecenter() {
    return recenter;
  }

  public PCA setRecenter(boolean recenter) {
    this.recenter = recenter;
    return this;
  }

  public boolean isRescale() {
    return rescale;
  }

  public PCA setRescale(boolean rescale) {
    this.rescale = rescale;
    return this;
  }

  public double getEigenvaluePower() {
    return eigenvaluePower;
  }

  public PCA setEigenvaluePower(double eigenvaluePower) {
    this.eigenvaluePower = eigenvaluePower;
    return this;
  }

}
