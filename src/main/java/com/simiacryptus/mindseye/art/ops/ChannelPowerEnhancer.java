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
import com.simiacryptus.mindseye.art.VisualModifierParameters;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.layers.cudnn.AvgReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.SquareActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.SumReducerLayer;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;

public @com.simiacryptus.ref.lang.RefAware
class ChannelPowerEnhancer implements VisualModifier {

  private boolean averaging = true;
  private boolean balanced = true;

  public boolean isAveraging() {
    return averaging;
  }

  public ChannelPowerEnhancer setAveraging(boolean averaging) {
    this.averaging = averaging;
    return this;
  }

  public boolean isBalanced() {
    return balanced;
  }

  public ChannelPowerEnhancer setBalanced(boolean balanced) {
    this.balanced = balanced;
    return this;
  }

  @Override
  public PipelineNetwork build(VisualModifierParameters visualModifierParameters) {
    PipelineNetwork network = visualModifierParameters.network;
    network = network.copyPipeline();
    double mag = balanced ? network.eval(visualModifierParameters.style).getData().get(0).rms() : 1;
    final Layer[] layers = new Layer[]{new SquareActivationLayer(),
        isAveraging() ? new AvgReducerLayer() : new SumReducerLayer(),
        new LinearActivationLayer().setScale(-Math.pow(mag, -2))};
    network.add(PipelineNetwork.build(1, layers).setName(String.format("-RMS / %.0E", mag))).freeRef();
    final PipelineNetwork freeze = (PipelineNetwork) network.freeze();
    visualModifierParameters.freeRef();
    return freeze;
  }
}
