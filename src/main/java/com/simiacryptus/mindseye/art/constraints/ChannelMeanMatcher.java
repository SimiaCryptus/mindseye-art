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

package com.simiacryptus.mindseye.art.constraints;

import com.simiacryptus.mindseye.art.VisualModifier;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;

public class ChannelMeanMatcher implements VisualModifier {

  private boolean balanced = true;
  private boolean averaging = true;

  @Override
  public PipelineNetwork build(PipelineNetwork network, Tensor image) {
    //network = network.copy();
    network.wrap(new BandAvgReducerLayer()).freeRef();
    Tensor result = network.eval(image).getDataAndFree().getAndFree(0);
    double mag = isBalanced() ? result.rms() : 1;
    network.wrap(PipelineNetwork.wrap(1,
        new ImgBandBiasLayer(result.scaleInPlace(-1)),
        new SquareActivationLayer(),
        isAveraging() ? new AvgReducerLayer() : new SumReducerLayer(),
        new NthPowerActivationLayer().setPower(0.5),
        new LinearActivationLayer().setScale(Math.pow(mag, -1))
    ).setName(String.format("RMS[x-C] / %.0E", mag))).freeRef();
    return (PipelineNetwork) network.freeze();
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
