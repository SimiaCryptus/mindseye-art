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

package com.simiacryptus.mindseye.art;

import com.simiacryptus.lang.ref.ReferenceCountingBase;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.util.ImageUtil;
import org.jetbrains.annotations.NotNull;

import java.util.function.UnaryOperator;

public class VisualModifierParameters extends ReferenceCountingBase {
  public final PipelineNetwork network;
  public final Tensor mask;
  public final UnaryOperator<Tensor> viewLayer;
  public final Tensor[] style;
  private final int[] contentDims;

  public VisualModifierParameters(PipelineNetwork network, int[] contentDims, UnaryOperator<Tensor> viewLayer, Tensor mask, Tensor... styleImages) {
    this.network = null == network ? network : network.addRef();
    this.mask = null == mask ? mask : mask.addRef();
    this.contentDims = contentDims;
    this.viewLayer = viewLayer;
    this.style = styleImages;
    for (Tensor tensor : this.style) {
      if (null != tensor) tensor.addRef();
    }
  }

  @Override
  public VisualModifierParameters addRef() {
    return (VisualModifierParameters) super.addRef();
  }

  @Override
  protected void _free() {
    if (null != network) network.freeRef();
    if (null != mask) mask.freeRef();
    for (Tensor tensor : this.style) {
      if (null != tensor) tensor.freeRef();
    }
    super._free();
  }

  @NotNull
  public VisualModifierParameters withMask(Tensor mask) {
    if (null != mask) {
      mask = Tensor.fromRGB(ImageUtil.resize(mask.toRgbImage(), contentDims[0], contentDims[1]));
      if (null != viewLayer) mask = viewLayer.apply(mask);
    }
    final VisualModifierParameters visualModifierParameters = new VisualModifierParameters(network, contentDims, viewLayer, mask, style);
    if (null != mask) mask.freeRef();
    this.freeRef();
    return visualModifierParameters;
  }

}
