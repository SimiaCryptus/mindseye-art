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
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;

public class RMSContentMatcher implements VisualModifier {

  @Override
  public PipelineNetwork build(PipelineNetwork original, Tensor image) {
    PipelineNetwork network = original.copy();
    Tensor baseContent = network.eval(image).getDataAndFree().getAndFree(0);
    double rms = Math.sqrt(baseContent.sumSq() / baseContent.length());
    DAGNode head = network.getHead();
    DAGNode constNode = network.constValueWrap(baseContent.scale(-1));
    Layer layer = original.getHead().getLayer();
    if(layer != null) constNode.getLayer().setName((layer != null?layer.getName():"Original") + " Content");
    network.wrap(new SumInputsLayer().setName("Difference"), head, constNode);
    network.wrap(PipelineNetwork.wrap(1,
        new SquareActivationLayer(),
        new AvgReducerLayer(),
        new NthPowerActivationLayer().setPower(0.5),
        new LinearActivationLayer().setScale(Math.pow(rms,-1))
    ).setName(String.format("-RMS / %.0E", rms)));
    return (PipelineNetwork) network.freeze();
  }

}
