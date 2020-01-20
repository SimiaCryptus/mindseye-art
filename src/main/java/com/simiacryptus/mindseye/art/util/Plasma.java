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

import javax.annotation.Nonnull;
import java.util.function.DoubleUnaryOperator;
import java.util.function.IntFunction;
import java.util.function.IntUnaryOperator;

public class Plasma {
  private int bands;
  private double[] noiseAmplitude;
  private double noisePower;

  public Plasma() {
    setNoisePower(2);
    setNoiseAmplitude(100);
    setBands(3);
  }

  public int getBands() {
    return bands;
  }

  @Nonnull
  public Plasma setBands(int bands) {
    this.bands = bands;
    return this;
  }

  public double[] getNoiseAmplitude() {
    return noiseAmplitude;
  }

  @Nonnull
  public Plasma setNoiseAmplitude(double... noiseAmplitude) {
    this.noiseAmplitude = noiseAmplitude;
    return this;
  }

  public double getNoisePower() {
    return noisePower;
  }

  @Nonnull
  public Plasma setNoisePower(double noisePower) {
    this.noisePower = noisePower;
    return this;
  }

  @Nonnull
  private static Tensor initSquare(final int bands) {
    Tensor tensor2 = new Tensor(1, 1, bands);
    tensor2.setByCoord(c1 -> 100 + 200 * (Math.random() - 0.5));
    Tensor baseColor = tensor2.addRef();
    Tensor tensor1 = new Tensor(2, 2, bands);
    tensor1.setByCoord(c -> baseColor.get(0, 0, c.getCoords()[2]));
    Tensor tensor = tensor1.addRef();
    baseColor.freeRef();
    return tensor;
  }

  @Nonnull
  public Tensor paint(final int width, final int height) {
    Tensor initSquare = initSquare(bands);
    Tensor expandPlasma = expandPlasma(initSquare, width, height);
    initSquare.freeRef();
    return expandPlasma;
  }

  @Nonnull
  private Tensor expandPlasma(@Nonnull Tensor image, final int width, final int height) {
    image.addRef();
    while (image.getDimensions()[0] < Math.max(width, height)) {
      final double factor = Math.pow(image.getDimensions()[0], noisePower);
      Tensor newImage = expandPlasma(image, RefArrays.stream(noiseAmplitude).map(v -> v / factor).toArray());
      image.freeRef();
      image = newImage;
    }
    Tensor tensor = Tensor.fromRGB(ImageUtil.resize(image.toRgbImage(), width, height));
    image.freeRef();
    return tensor;
  }

  @Nonnull
  private Tensor expandPlasma(@Nonnull final Tensor seed, @Nonnull double... noise) {
    int bands = seed.getDimensions()[2];
    int width = seed.getDimensions()[0] * 2;
    int height = seed.getDimensions()[1] * 2;
    Tensor returnValue = new Tensor(width, height, bands);
    IntFunction<DoubleUnaryOperator> fn1 = b -> x -> Math
        .max(Math.min(x + noise[b % noise.length] * (Math.random() - 0.5), 255), 0);
    IntFunction<DoubleUnaryOperator> fn2 = b -> x -> Math
        .max(Math.min(x + Math.sqrt(2) * noise[b % noise.length] * (Math.random() - 0.5), 255), 0);
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
          double value = (returnValue.get(addrX.applyAsInt(x - 1), addrY.applyAsInt(y - 1), band))
              + (returnValue.get(addrX.applyAsInt(x - 1), addrY.applyAsInt(y + 1), band))
              + (returnValue.get(addrX.applyAsInt(x + 1), addrY.applyAsInt(y - 1), band))
              + (returnValue.get(addrX.applyAsInt(x + 1), addrY.applyAsInt(y + 1), band));
          value = f2_band.applyAsDouble(value / 4);
          returnValue.set(x, y, band, value);
        }
      }
      final DoubleUnaryOperator f1_band = fn1.apply(band);
      for (int x = 0; x < width; x += 2) {
        for (int y = 1; y < height; y += 2) {
          double value = (returnValue.get(addrX.applyAsInt(x - 1), addrY.applyAsInt(y), band))
              + (returnValue.get(addrX.applyAsInt(x + 1), addrY.applyAsInt(y), band))
              + (returnValue.get(addrX.applyAsInt(x), addrY.applyAsInt(y - 1), band))
              + (returnValue.get(addrX.applyAsInt(x), addrY.applyAsInt(y + 1), band));
          value = f1_band.applyAsDouble(value / 4);
          returnValue.set(x, y, band, value);
        }
      }
      for (int x = 1; x < width; x += 2) {
        for (int y = 0; y < height; y += 2) {
          double value = (returnValue.get(addrX.applyAsInt(x - 1), addrY.applyAsInt(y), band))
              + (returnValue.get(addrX.applyAsInt(x + 1), addrY.applyAsInt(y), band))
              + (returnValue.get(addrX.applyAsInt(x), addrY.applyAsInt(y - 1), band))
              + (returnValue.get(addrX.applyAsInt(x), addrY.applyAsInt(y + 1), band));
          value = f1_band.applyAsDouble(value / 4);
          returnValue.set(x, y, band, value);
        }
      }
    }
    return returnValue;
  }
}
