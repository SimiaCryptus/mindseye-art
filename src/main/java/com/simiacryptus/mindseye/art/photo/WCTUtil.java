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

public class WCTUtil {

  public static PipelineNetwork applicator(Tensor encodedStyle) {
    return applicator(encodedStyle, normalizer());
  }

  public static PipelineNetwork applicator(Tensor encodedStyle, Layer normalNetwork) {
    final Tensor meanSignal = means(encodedStyle);
    final Tensor signalPower = rms(encodedStyle, meanSignal);
    final PipelineNetwork applicator = applicator(meanSignal, signalPower, normalNetwork);
    meanSignal.freeRef();
    signalPower.freeRef();
    return applicator;
  }

  public static PipelineNetwork applicator(Tensor meanSignal, Tensor signalPower, Layer normalNetwork) {
    final PipelineNetwork applicator = new PipelineNetwork(1);
    applicator.wrap(
        new ImgBandBiasLayer(meanSignal),
        applicator.wrap(
            new ProductLayer(),
            applicator.getInput(0),
            applicator.constValue(signalPower)
        )
    ).freeRef();
    return PipelineNetwork.wrap(1, normalNetwork, applicator);
  }

  public static Layer normalizer() {
    final PipelineNetwork normalizer = new PipelineNetwork(1);
    final InnerNode avgNode = normalizer.wrap(new BandAvgReducerLayer());
    final InnerNode centered = normalizer.wrap(
        new ImgBandDynamicBiasLayer(),
        normalizer.getInput(0),
        normalizer.wrap(new ScaleLayer(-1), avgNode)
    );
    final InnerNode scales = normalizer.wrap(
        PipelineNetwork.wrap(1,
            new SquareActivationLayer(),
            new BandAvgReducerLayer(),
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
}
