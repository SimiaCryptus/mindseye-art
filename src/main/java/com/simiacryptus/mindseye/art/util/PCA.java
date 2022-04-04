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

import com.simiacryptus.mindseye.lang.Result;
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

/**
 * The type Pca.
 */
public class PCA {
  private boolean recenter;
  private boolean rescale;
  private double eigenvaluePower;

  /**
   * Instantiates a new Pca.
   *
   * @param recenter        the recenter
   * @param rescale         the rescale
   * @param eigenvaluePower the eigenvalue power
   */
  public PCA(boolean recenter, boolean rescale, double eigenvaluePower) {
    this.setRecenter(recenter);
    this.setRescale(rescale);
    this.setEigenvaluePower(eigenvaluePower);
  }

  /**
   * Instantiates a new Pca.
   */
  public PCA() {
    this(true, false, 0.0);
  }

  /**
   * Gets eigenvalue power.
   *
   * @return the eigenvalue power
   */
  public double getEigenvaluePower() {
    return eigenvaluePower;
  }

  /**
   * Sets eigenvalue power.
   *
   * @param eigenvaluePower the eigenvalue power
   * @return the eigenvalue power
   */
  @Nonnull
  public PCA setEigenvaluePower(double eigenvaluePower) {
    this.eigenvaluePower = eigenvaluePower;
    return this;
  }

  /**
   * Is recenter boolean.
   *
   * @return the boolean
   */
  public boolean isRecenter() {
    return recenter;
  }

  /**
   * Sets recenter.
   *
   * @param recenter the recenter
   * @return the recenter
   */
  @Nonnull
  public PCA setRecenter(boolean recenter) {
    this.recenter = recenter;
    return this;
  }

  /**
   * Is rescale boolean.
   *
   * @return the boolean
   */
  public boolean isRescale() {
    return rescale;
  }

  /**
   * Sets rescale.
   *
   * @param rescale the rescale
   * @return the rescale
   */
  @Nonnull
  public PCA setRescale(boolean rescale) {
    this.rescale = rescale;
    return this;
  }

  /**
   * Band covariance double [ ].
   *
   * @param pixelStream the pixel stream
   * @param pixels      the pixels
   * @param mean        the mean
   * @param rms         the rms
   * @return the double [ ]
   */
  public static double[] bandCovariance(@Nonnull final RefStream<double[]> pixelStream, final int pixels, final double[] mean,
                                        final double[] rms) {
    return RefArrays.stream(RefUtil.get(pixelStream.map(pixel -> {
      double[] crossproduct = RecycleBin.DOUBLES.obtain(pixel.length * pixel.length);
      int k = 0;
      for (int j = 0; j < pixel.length; j++) {
        for (int i = 0; i < pixel.length; i++) {
          crossproduct[k++] = (pixel[i] - mean[i]) / (rms[i] == 0 ? 1 : rms[i])
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

  /**
   * Count pixels int.
   *
   * @param featureImage the feature image
   * @return the int
   */
  public static int countPixels(@Nonnull final Tensor featureImage) {
    int[] dimensions = featureImage.getDimensions();
    featureImage.freeRef();
    int width = dimensions[0];
    int height = dimensions[1];
    return width * height;
  }

  /**
   * Pca ref list.
   *
   * @param bandCovariance the band covariance
   * @param eigenPower     the eigen power
   * @return the ref list
   */
  public static RefList<Tensor> pca(@Nonnull final double[] bandCovariance, final double eigenPower) {
    @Nonnull final EigenDecomposition decomposition = new EigenDecomposition(toMatrix(bandCovariance));
    return RefIntStream.range(0, (int) Math.sqrt(bandCovariance.length)).mapToObj(vectorIndex -> {
      double[] data = decomposition.getEigenvector(vectorIndex).toArray();
      Tensor tensor = new Tensor(data, 1, 1, data.length).unit();
      tensor.scaleInPlace(Math.pow(decomposition.getRealEigenvalue(vectorIndex), eigenPower));
      return tensor;
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

  /**
   * Channel pca ref list.
   *
   * @param image the image
   * @return the ref list
   */
  public RefList<Tensor> channelPCA(@Nonnull Tensor image) {
    Tensor meanTensor = getChannelMeans(image.addRef());
    Tensor scaled = getChannelRms(image.addRef(), image.getDimensions()[2], meanTensor.addRef());
    assert scaled != null;
    double[] bandCovariance = bandCovariance(image.getPixelStream(), countPixels(image.addRef()), meanTensor.getData(),
        scaled.getData());
    image.freeRef();
    meanTensor.freeRef();
    scaled.freeRef();
    return pca(bandCovariance, getEigenvaluePower());
  }

  /**
   *
   * public Tensor getChannelRms(Tensor image, int bands, @Nonnull Tensor meanTensor)
   *
   * This code calculates the root mean square (RMS) of an image, using a mean tensor.
   * If the image does not need to be rescaled, the mean tensor is simply returned.
   * Otherwise, a pipeline network is used to apply a square activation layer, a band reducer layer, and an Nth power activation layer (with a power of 0.5) to the image.
   * The resulting data is then mapped so that any zeros are replaced with ones.
   *
   * @param image The image to be processed.
   * @param bands The number of bands in the image.
   * @param meanTensor The mean tensor.
   * @return The RMS of the image.
   */
  @Nullable
  public Tensor getChannelRms(Tensor image, int bands, @Nonnull Tensor meanTensor) {
    if (!isRescale()) {
      Tensor tensor = meanTensor.map(x -> 1);
      meanTensor.freeRef();
      image.freeRef();
      return tensor;
    } else {
      NthPowerActivationLayer nthPowerActivationLayer = new NthPowerActivationLayer();
      nthPowerActivationLayer.setPower(0.5);
      BandReducerLayer bandReducerLayer = new BandReducerLayer();
      bandReducerLayer.setMode(PoolingLayer.PoolingMode.Avg);
      ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(bands);
      imgBandBiasLayer.set(meanTensor.scale(-1));
      meanTensor.freeRef();
      PipelineNetwork network = PipelineNetwork.build(1,
          imgBandBiasLayer,
          new SquareActivationLayer(),
          bandReducerLayer,
          nthPowerActivationLayer
      );
      try {
        Tensor data0 = Result.getData0(network.eval(image));
        Tensor map = data0.map(x -> x == 0.0 ? 1.0 : x);
        data0.freeRef();
        return map;
      } finally {
        network.freeRef();
      }
    }
  }

  /**
   * Gets channel means.
   *
   * @param image the image
   * @return the channel means
   */
  @Nonnull
  public Tensor getChannelMeans(Tensor image) {
    BandReducerLayer bandReducerLayer = new BandReducerLayer();
    bandReducerLayer.setMode(PoolingLayer.PoolingMode.Avg);
    Tensor meanTensor = Result.getData0(bandReducerLayer.eval(image));
    bandReducerLayer.freeRef();
    if (!isRecenter())
      meanTensor.fill(0);
    return meanTensor;
  }


}
