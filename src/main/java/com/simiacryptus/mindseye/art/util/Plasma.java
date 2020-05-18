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
import com.simiacryptus.mindseye.util.ImageUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.util.data.DoubleStatistics;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.util.function.DoubleUnaryOperator;
import java.util.function.IntFunction;
import java.util.function.IntUnaryOperator;
import java.util.stream.IntStream;

/**
 * The type Plasma.
 */
public class Plasma {
  private int bands;
  private double[] noiseAmplitude;
  private double noisePower;

  /**
   * Instantiates a new Plasma.
   */
  public Plasma() {
    setNoisePower(0.5);
    setNoiseAmplitude(100);
    setBands(3);
  }

  /**
   * Gets bands.
   *
   * @return the bands
   */
  public int getBands() {
    return bands;
  }

  /**
   * Sets bands.
   *
   * @param bands the bands
   * @return the bands
   */
  @Nonnull
  public Plasma setBands(int bands) {
    this.bands = bands;
    return this;
  }

  /**
   * Get noise amplitude double [ ].
   *
   * @return the double [ ]
   */
  public double[] getNoiseAmplitude() {
    return noiseAmplitude;
  }

  /**
   * Sets noise amplitude.
   *
   * @param noiseAmplitude the noise amplitude
   * @return the noise amplitude
   */
  @Nonnull
  public Plasma setNoiseAmplitude(double... noiseAmplitude) {
    this.noiseAmplitude = noiseAmplitude;
    return this;
  }

  /**
   * Gets noise power.
   *
   * @return the noise power
   */
  public double getNoisePower() {
    return noisePower;
  }

  /**
   * Sets noise power.
   *
   * @param noisePower the noise power
   * @return the noise power
   */
  @Nonnull
  public Plasma setNoisePower(double noisePower) {
    this.noisePower = noisePower;
    return this;
  }

  /**
   * Band stats double [ ] [ ].
   *
   * @param image the image
   * @return the double [ ] [ ]
   */
  public static double[][] bandStats(Tensor image) {
    double[][] doubles = IntStream.range(0, image.getDimensions()[2]).mapToObj(band -> {
      Tensor selectBand = image.selectBand(band);
      DoubleStatistics doubleStatistics = selectBand.getDoubleStatistics();
      selectBand.freeRef();
      double min = doubleStatistics.getMin();
      double scale = 255 / (doubleStatistics.getMax() - min);
      return new double[]{min, scale};
    }).toArray(double[][]::new);
    image.freeRef();
    return doubles;
  }

  /**
   * To image buffered image.
   *
   * @param image the image
   * @return the buffered image
   */
  @NotNull
  public static BufferedImage toImage(@Nonnull Tensor image) {
    BufferedImage rgbImage = image.toRgbImage();
    image.freeRef();
    return rgbImage;
  }

  /**
   * Resize tensor.
   *
   * @param rgbImage the rgb image
   * @param width    the width
   * @param height   the height
   * @return the tensor
   */
  @NotNull
  public static Tensor resize(BufferedImage rgbImage, int width, int height) {
    return Tensor.fromRGB(ImageUtil.resize(rgbImage, width, height));
  }

  @Nonnull
  private static Tensor initSquare(final int bands) {
    Tensor tensor = new Tensor(1, 1, bands);
    tensor.setByCoord(c1 -> 100 + 200 * (Math.random() - 0.5));
    Tensor tensor1 = new Tensor(2, 2, bands);
    tensor1.setByCoord(c -> tensor.get(0, 0, c.getCoords()[2]));
    tensor.freeRef();
    return tensor1;
  }

  /**
   * Paint tensor.
   *
   * @param width  the width
   * @param height the height
   * @return the tensor
   */
  @Nonnull
  public Tensor paint(final int width, final int height) {
    Tensor image = initSquare(bands);
    while (image.getDimensions()[0] < Math.max(width, height)) {
      final double factor = Math.pow(image.getDimensions()[0], noisePower);
      Tensor newImage = expandPlasma(image.addRef(), RefArrays.stream(noiseAmplitude).map(v -> v / factor).toArray());
      if (image != newImage) {
        image.freeRef();
        image = newImage;
      } else {
        newImage.freeRef();
      }
    }
    return resize(toImage(renormBands(image)), width, height);
  }

  /**
   * Renorm bands tensor.
   *
   * @param image the image
   * @return the tensor
   */
  @NotNull
  public Tensor renormBands(Tensor image) {
    double[][] bandStats = bandStats(image.addRef());
    Tensor mapCoords = image.mapCoords(c -> {
      int band = c.getCoords()[2];
      return (image.get(c) - bandStats[band][0]) * bandStats[band][1];
    });
    image.freeRef();
    return mapCoords;
  }

  @Nonnull
  private Tensor expandPlasma(@Nonnull final Tensor seed, @Nonnull double... noise) {
    int bands = seed.getDimensions()[2];
    int width = seed.getDimensions()[0] * 2;
    int height = seed.getDimensions()[1] * 2;
    Tensor returnValue = new Tensor(width, height, bands);
    IntFunction<DoubleUnaryOperator> fn1 = b -> x -> clamp(x + noise[b % noise.length] * (Math.random() - 0.5));
    IntFunction<DoubleUnaryOperator> fn2 = b -> x -> clamp(x + Math.sqrt(2) * noise[b % noise.length] * (Math.random() - 0.5));
    IntUnaryOperator addrX = x -> {
      while (x >= width)
        x -= width;
      while (x < 0)
        x += width;
      return x;
    };
    IntUnaryOperator addrY = x -> {
      while (x >= height)
        x -= height;
      while (x < 0)
        x += height;
      return x;
    };
    for (int band = 0; band < bands; band++) {
      for (int x = 0; x < width; x += 2) {
        for (int y = 0; y < height; y += 2) {
          double value = seed.get(x / 2, y / 2, band);
          returnValue.set(x, y, band, value);
        }
      }
      final DoubleUnaryOperator f2_band = fn2.apply(band);
      for (int x = 1; x < width; x += 2) {
        for (int y = 1; y < height; y += 2) {
          double value = returnValue.get(addrX.applyAsInt(x - 1), addrY.applyAsInt(y - 1), band)
              + returnValue.get(addrX.applyAsInt(x - 1), addrY.applyAsInt(y + 1), band)
              + returnValue.get(addrX.applyAsInt(x + 1), addrY.applyAsInt(y - 1), band)
              + returnValue.get(addrX.applyAsInt(x + 1), addrY.applyAsInt(y + 1), band);
          value = f2_band.applyAsDouble(value / 4);
          returnValue.set(x, y, band, value);
        }
      }
      final DoubleUnaryOperator f1_band = fn1.apply(band);
      for (int x = 0; x < width; x += 2) {
        for (int y = 1; y < height; y += 2) {
          double value = returnValue.get(addrX.applyAsInt(x - 1), addrY.applyAsInt(y), band)
              + returnValue.get(addrX.applyAsInt(x + 1), addrY.applyAsInt(y), band)
              + returnValue.get(addrX.applyAsInt(x), addrY.applyAsInt(y - 1), band)
              + returnValue.get(addrX.applyAsInt(x), addrY.applyAsInt(y + 1), band);
          value = f1_band.applyAsDouble(value / 4);
          returnValue.set(x, y, band, value);
        }
      }
      for (int x = 1; x < width; x += 2) {
        for (int y = 0; y < height; y += 2) {
          double value = returnValue.get(addrX.applyAsInt(x - 1), addrY.applyAsInt(y), band)
              + returnValue.get(addrX.applyAsInt(x + 1), addrY.applyAsInt(y), band)
              + returnValue.get(addrX.applyAsInt(x), addrY.applyAsInt(y - 1), band)
              + returnValue.get(addrX.applyAsInt(x), addrY.applyAsInt(y + 1), band);
          value = f1_band.applyAsDouble(value / 4);
          returnValue.set(x, y, band, value);
        }
      }
    }
    seed.freeRef();
    return returnValue;
  }

  private double clamp(double a) {
    return Math.max(Math.min(a, 255), 0);
  }
}
