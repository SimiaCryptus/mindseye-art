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

package com.simiacryptus.mindseye.art.ops;

import com.simiacryptus.mindseye.art.VisualModifier;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;

public class ChannelMeanMatcher implements VisualModifier {

  private boolean balanced = true;
  private boolean averaging = true;

  @Override
  public PipelineNetwork build(PipelineNetwork network, Tensor... image) {
    Tensor meanSignal = null;
    return buildWithModel(network, meanSignal, image);
  }

  @NotNull
  public PipelineNetwork buildWithModel(PipelineNetwork network, Tensor meanSignal, Tensor... image) {
    network = network.copyPipeline();
    network.wrap(new BandAvgReducerLayer()).freeRef();
    if (meanSignal == null) meanSignal = channelMeans(network, image);
    double mag = isBalanced() ? meanSignal.rms() : 1;
    network.wrap(PipelineNetwork.wrap(1,
        new ImgBandBiasLayer(meanSignal.scaleInPlace(-1)),
        new SquareActivationLayer(),
        isAveraging() ? new AvgReducerLayer() : new SumReducerLayer(),
        new LinearActivationLayer().setScale(Math.pow(mag, -2))
//        ,new NthPowerActivationLayer().setPower(0.5)
    ).setName(String.format("RMS[x-C] / %.0E", mag))).freeRef();
    return (PipelineNetwork) network.freeze();
  }

  @NotNull
  private Tensor channelMeans(PipelineNetwork finalNetwork, Tensor... image) {
    return Arrays.stream(image).map(tensor ->
        finalNetwork.eval(tensor).getDataAndFree().getAndFree(0)
    ).reduce((a, b) -> {
      Tensor c = a.addAndFree(b);
      b.freeRef();
      return c;
    }).get().scaleInPlace(1.0 / image.length);
  }

  public boolean isBalanced() {
    return balanced;
  }

  public ChannelMeanMatcher setBalanced(boolean balanced) {
    this.balanced = balanced;
    return this;
  }

  public boolean isAveraging() {
    return averaging;
  }

  public ChannelMeanMatcher setAveraging(boolean averaging) {
    this.averaging = averaging;
    return this;
  }
}
