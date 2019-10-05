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

package com.simiacryptus.mindseye.art.photo;

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer;
import com.simiacryptus.mindseye.network.InnerNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import org.jetbrains.annotations.NotNull;

public class WCTUtil {

  public static PipelineNetwork applicator(Tensor encodedStyle, double contentDensity, double styleDensity) {
    return PipelineNetwork.wrap(1, normalizer(contentDensity), renormalizer(encodedStyle, styleDensity));
  }

  @NotNull
  public static PipelineNetwork renormalizer(Tensor encodedStyle, double styleDensity) {
    final Tensor meanSignal = means(encodedStyle).scaleInPlace(1.0 / styleDensity);
    final Tensor signalPower = rms(encodedStyle, meanSignal).scaleInPlace(Math.sqrt(1.0 / styleDensity));
    final PipelineNetwork renormalizer = new PipelineNetwork(1);
    renormalizer.wrap(
        new ImgBandBiasLayer(meanSignal),
        renormalizer.wrap(
            new ProductLayer(),
            renormalizer.getInput(0),
            renormalizer.constValue(signalPower)
        )
    ).freeRef();
    meanSignal.freeRef();
    signalPower.freeRef();
    return renormalizer;
  }

  public static Layer normalizer(double maskFactor) {
    final PipelineNetwork normalizer = new PipelineNetwork(1);
    final InnerNode avgNode = normalizer.wrap(new BandAvgReducerLayer());
    final InnerNode centered = normalizer.wrap(
        new ImgBandDynamicBiasLayer(),
        normalizer.getInput(0),
        normalizer.wrap(new ScaleLayer(-1 / maskFactor), avgNode)
    );
    final InnerNode scales = normalizer.wrap(
        PipelineNetwork.wrap(1,
            new SquareActivationLayer(),
            new BandAvgReducerLayer(),
            new ScaleLayer(1 / maskFactor),
            new NthPowerActivationLayer().setPower(-0.5)
        ),
        centered.addRef()
    );
    final InnerNode rescaled = normalizer.wrap(
        new ProductLayer(),
        centered,
        scales
    );
    rescaled.freeRef();
    return normalizer.freeze();
  }

  public static Tensor means(Tensor encodedStyle) {
    final BandAvgReducerLayer avgReducerLayer = new BandAvgReducerLayer();
    final Tensor tensor = avgReducerLayer.eval(encodedStyle).getDataAndFree().getAndFree(0);
    avgReducerLayer.freeRef();
    return tensor;
  }

  public static Tensor rms(Tensor normalFeatures, Tensor normalMeanSignal) {
    final Tensor scale = normalMeanSignal.scale(-1);
    final PipelineNetwork wrap = PipelineNetwork.wrap(1,
        new ImgBandBiasLayer(scale),
        new SquareActivationLayer(),
        new BandAvgReducerLayer(),
        new NthPowerActivationLayer().setPower(0.5)
    );
    scale.freeRef();
    final Tensor tensor = wrap.eval(normalFeatures).getDataAndFree().getAndFree(0);
    wrap.freeRef();
    return tensor;
  }

  public static Layer normalizer() {
    return normalizer(1.0);
  }

  public static PipelineNetwork applicator(Tensor encodedStyle) {
    return applicator(encodedStyle, 1.0, 1.0);
  }
}
