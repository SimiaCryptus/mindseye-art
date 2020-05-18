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
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefString;

import javax.annotation.Nonnull;

/**
 * The type Content matcher.
 */
public class ContentMatcher implements VisualModifier {

  private boolean averaging = true;
  private boolean balanced = true;

  /**
   * Is averaging boolean.
   *
   * @return the boolean
   */
  public boolean isAveraging() {
    return averaging;
  }

  /**
   * Sets averaging.
   *
   * @param averaging the averaging
   * @return the averaging
   */
  @Nonnull
  public ContentMatcher setAveraging(boolean averaging) {
    this.averaging = averaging;
    return this;
  }

  /**
   * Is balanced boolean.
   *
   * @return the boolean
   */
  public boolean isBalanced() {
    return balanced;
  }

  /**
   * Sets balanced.
   *
   * @param balanced the balanced
   * @return the balanced
   */
  @Nonnull
  public ContentMatcher setBalanced(boolean balanced) {
    this.balanced = balanced;
    return this;
  }

  /**
   * Sets name.
   *
   * @param constNode the const node
   * @param name      the name
   */
  public static void setName(DAGNode constNode, String name) {
    Layer layer = constNode.getLayer();
    constNode.freeRef();
    layer.setName(name);
    layer.freeRef();
  }

  @Nonnull
  @Override
  public PipelineNetwork build(@Nonnull VisualModifierParameters visualModifierParameters) {
    Tensor[] style = visualModifierParameters.getStyle();
    if (1 != style.length) {
      RefUtil.freeRef(style);
      visualModifierParameters.freeRef();
      throw new IllegalArgumentException();
    }
    PipelineNetwork network = visualModifierParameters.copyNetwork();
    DAGNode networkHead = network.getHead();
    Layer layer = networkHead.getLayer();
    networkHead.freeRef();
    String name = (layer != null ? layer.getName() : "Original") + " Content";
    layer.freeRef();

    Tensor mask = visualModifierParameters.getMask();
    if (mask != null) {
      network.add(new ProductLayer(),
          network.getHead(),
          network.constValue(
              MomentMatcher.toMask(MomentMatcher.transform(network.addRef(), mask, Precision.Float))
          )
      ).freeRef();
    }

    Tensor baseContent = Result.getData0(network.eval(style));
    visualModifierParameters.freeRef();
    double mag = balanced ? baseContent.rms() : 1;
    if (!Double.isFinite(mag) || mag < 0) {
      baseContent.freeRef();
      network.freeRef();
      throw new RuntimeException("RMS = " + mag);
    }
    DAGNode head = network.getHead();
    baseContent.scaleInPlace(-1);
    DAGNode constNode = network.constValueWrap(baseContent);
    assert constNode != null;
    setName(constNode.addRef(), name);
    Layer layer2 = new SumInputsLayer();
    layer2.setName("Difference");
    network.add(layer2, head, constNode).freeRef();
    LinearActivationLayer linearActivationLayer = new LinearActivationLayer();
    final double scale = 0 == mag ? 1 : Math.pow(mag, -1);
    linearActivationLayer.setScale(scale);
    Layer layer1 = PipelineNetwork.build(1,
        linearActivationLayer,
        new SquareActivationLayer(),
        isAveraging() ? new AvgReducerLayer() : new SumReducerLayer()
    );
    layer1.setName(RefString.format("RMS / %.0E", mag));
    network.add(layer1).freeRef();
    network.freeze();
    return network;
  }
}
