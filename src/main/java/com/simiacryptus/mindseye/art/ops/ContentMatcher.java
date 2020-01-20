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
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.wrappers.RefString;

import javax.annotation.Nonnull;

public class ContentMatcher implements VisualModifier {

  private boolean averaging = true;
  private boolean balanced = true;

  public boolean isAveraging() {
    return averaging;
  }

  @Nonnull
  public ContentMatcher setAveraging(boolean averaging) {
    this.averaging = averaging;
    return this;
  }

  public boolean isBalanced() {
    return balanced;
  }

  @Nonnull
  public ContentMatcher setBalanced(boolean balanced) {
    this.balanced = balanced;
    return this;
  }

  @Nonnull
  @Override
  public PipelineNetwork build(@Nonnull VisualModifierParameters visualModifierParameters) {
    if (1 != visualModifierParameters.style.length)
      throw new IllegalArgumentException();
    PipelineNetwork network = visualModifierParameters.network;
    assert network != null;
    network = network.copyPipeline();
    assert network != null;
    Layer layer = network.getHead().getLayer();
    String name = (layer != null ? layer.getName() : "Original") + " Content";

    final Tensor boolMask = MomentMatcher
        .toMask(MomentMatcher.transform(network, visualModifierParameters.mask, Precision.Float));
    network.add(new ProductLayer(), network.getHead(), network.constValue(boolMask)).freeRef();

    Tensor baseContent = network.eval(visualModifierParameters.style).getData().get(0);
    visualModifierParameters.freeRef();
    double mag = balanced ? baseContent.rms() : 1;
    if (!Double.isFinite(mag) || mag < 0)
      throw new RuntimeException("RMS = " + mag);
    DAGNode head = network.getHead();
    baseContent.scaleInPlace(-1);
    DAGNode constNode = network.constValueWrap(baseContent.addRef());
    assert constNode != null;
    constNode.getLayer().setName(name);
    Layer layer2 = new SumInputsLayer();
    layer2.setName("Difference");
    network.add(layer2.addRef(), head, constNode).freeRef();
    LinearActivationLayer linearActivationLayer = new LinearActivationLayer();
    final double scale = 0 == mag ? 1 : Math.pow(mag, -1);
    linearActivationLayer.setScale(scale);
    final Layer[] layers = new Layer[]{linearActivationLayer.addRef(),
        new SquareActivationLayer(), isAveraging() ? new AvgReducerLayer() : new SumReducerLayer()};
    Layer layer1 = PipelineNetwork.build(1, layers);
    layer1.setName(RefString.format("RMS / %.0E", mag));
    network.add(layer1.addRef()).freeRef();
    network.freeze();
    return network.addRef();
  }
}
