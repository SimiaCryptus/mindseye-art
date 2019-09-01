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
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.AvgReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.SquareActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.SumInputsLayer;
import com.simiacryptus.mindseye.layers.cudnn.SumReducerLayer;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;

public class ContentMatcher implements VisualModifier {

  private boolean averaging = true;
  private boolean balanced = true;

  @Override
  public PipelineNetwork build(PipelineNetwork network, Tensor... image) {
    if (1 != image.length) throw new IllegalArgumentException();
    network = network.copyPipeline();
    Layer layer = network.getHead().getLayer();
    String name = (layer != null ? layer.getName() : "Original") + " Content";
    Tensor baseContent = network.eval(image).getDataAndFree().getAndFree(0);
    double mag = balanced ? baseContent.rms() : 1;
    if(!Double.isFinite(mag) || mag < 0) throw new RuntimeException("RMS = " + mag);
    DAGNode head = network.getHead();
    DAGNode constNode = network.constValueWrap(baseContent.scaleInPlace(-1));
    constNode.getLayer().setName(name);
    network.wrap(new SumInputsLayer().setName("Difference"), head, constNode).freeRef();
    network.wrap(PipelineNetwork.wrap(1,
        new LinearActivationLayer().setScale(0 == mag ? 1 : Math.pow(mag, -1)),
        new SquareActivationLayer(),
        isAveraging() ? new AvgReducerLayer() : new SumReducerLayer()
    ).setName(String.format("RMS / %.0E", mag))).freeRef();
    return (PipelineNetwork) network.freeze();
  }

  public boolean isAveraging() {
    return averaging;
  }

  public ContentMatcher setAveraging(boolean averaging) {
    this.averaging = averaging;
    return this;
  }

  public boolean isBalanced() {
    return balanced;
  }

  public ContentMatcher setBalanced(boolean balanced) {
    this.balanced = balanced;
    return this;
  }
}
