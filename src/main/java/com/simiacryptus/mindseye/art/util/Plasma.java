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

import javax.annotation.Nonnull;
import java.util.function.DoubleUnaryOperator;
import java.util.function.IntUnaryOperator;

public class Plasma {
  private int bands;
  private double noiseAmplitude;
  private double noisePower;

  public Plasma() {
    setNoisePower(2);
    setNoiseAmplitude(100);
    setBands(3);
  }

  @Nonnull
  private static Tensor initSquare(final int bands) {
    Tensor baseColor = new Tensor(1, 1, bands).setByCoord(c -> 100 + 200 * (Math.random() - 0.5));
    Tensor tensor = new Tensor(2, 2, bands).setByCoord(c -> baseColor.get(0, 0, c.getCoords()[2]));
    baseColor.freeRef();
    return tensor;
  }

  public Tensor paint(final int width, final int height) {
    Tensor initSquare = initSquare(bands);
    Tensor expandPlasma = expandPlasma(initSquare, width, height);
    initSquare.freeRef();
    return expandPlasma;
  }

  @Nonnull
  private Tensor expandPlasma(Tensor image, final int width, final int height) {
    image.addRef();
    while (image.getDimensions()[0] < Math.max(width, height)) {
      Tensor newImage = expandPlasma(image, Math.pow(noiseAmplitude / image.getDimensions()[0], noisePower));
      image.freeRef();
      image = newImage;
    }
    Tensor tensor = Tensor.fromRGB(ImageUtil.resize(image.toImage(), width, height));
    image.freeRef();
    return tensor;
  }

  private Tensor expandPlasma(final Tensor seed, double noise) {
    int bands = seed.getDimensions()[2];
    int width = seed.getDimensions()[0] * 2;
    int height = seed.getDimensions()[1] * 2;
    Tensor returnValue = new Tensor(width, height, bands);
    DoubleUnaryOperator fn1 = x -> Math.max(Math.min(x + noise * (Math.random() - 0.5), 255), 0);
    DoubleUnaryOperator fn2 = x -> Math.max(Math.min(x + Math.sqrt(2) * noise * (Math.random() - 0.5), 255), 0);
    IntUnaryOperator addrX = x -> {
      while (x >= width) x -= width;
      while (x < 0) x += width;
      return x;
    };
    IntUnaryOperator addrY = x -> {
      while (x >= height) x -= height;
      while (x < 0) x += height;
      return x;
    };
    for (int band = 0; band < bands; band++) {
      for (int x = 0; x < width; x += 2) {
        for (int y = 0; y < height; y += 2) {
          double value = seed.get(x / 2, y / 2, band);
          returnValue.set(x, y, band, value);
        }
      }
      for (int x = 1; x < width; x += 2) {
        for (int y = 1; y < height; y += 2) {
          double value = (returnValue.get(addrX.applyAsInt(x - 1), addrY.applyAsInt(y - 1), band)) +
              (returnValue.get(addrX.applyAsInt(x - 1), addrY.applyAsInt(y + 1), band)) +
              (returnValue.get(addrX.applyAsInt(x + 1), addrY.applyAsInt(y - 1), band)) +
              (returnValue.get(addrX.applyAsInt(x + 1), addrY.applyAsInt(y + 1), band));
          value = fn2.applyAsDouble(value / 4);
          returnValue.set(x, y, band, value);
        }
      }
      for (int x = 0; x < width; x += 2) {
        for (int y = 1; y < height; y += 2) {
          double value = (returnValue.get(addrX.applyAsInt(x - 1), addrY.applyAsInt(y), band)) +
              (returnValue.get(addrX.applyAsInt(x + 1), addrY.applyAsInt(y), band)) +
              (returnValue.get(addrX.applyAsInt(x), addrY.applyAsInt(y - 1), band)) +
              (returnValue.get(addrX.applyAsInt(x), addrY.applyAsInt(y + 1), band));
          value = fn1.applyAsDouble(value / 4);
          returnValue.set(x, y, band, value);
        }
      }
      for (int x = 1; x < width; x += 2) {
        for (int y = 0; y < height; y += 2) {
          double value = (returnValue.get(addrX.applyAsInt(x - 1), addrY.applyAsInt(y), band)) +
              (returnValue.get(addrX.applyAsInt(x + 1), addrY.applyAsInt(y), band)) +
              (returnValue.get(addrX.applyAsInt(x), addrY.applyAsInt(y - 1), band)) +
              (returnValue.get(addrX.applyAsInt(x), addrY.applyAsInt(y + 1), band));
          value = fn1.applyAsDouble(value / 4);
          returnValue.set(x, y, band, value);
        }
      }
    }
    return returnValue;
  }

  public int getBands() {
    return bands;
  }

  public Plasma setBands(int bands) {
    this.bands = bands;
    return this;
  }

  public double getNoiseAmplitude() {
    return noiseAmplitude;
  }

  public Plasma setNoiseAmplitude(double noiseAmplitude) {
    this.noiseAmplitude = noiseAmplitude;
    return this;
  }

  public double getNoisePower() {
    return noisePower;
  }

  public Plasma setNoisePower(double noisePower) {
    this.noisePower = noisePower;
    return this;
  }
}
