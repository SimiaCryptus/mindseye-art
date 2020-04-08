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

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.util.ImageUtil;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCountingBase;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.awt.image.BufferedImage;
import java.util.function.UnaryOperator;

public class VisualModifierParameters extends ReferenceCountingBase {
  public final UnaryOperator<Tensor> viewLayer;
  @Nullable
  private final PipelineNetwork network;
  @Nonnull
  private final Tensor mask;
  private final Tensor[] style;
  private final int[] contentDims;

  public VisualModifierParameters(@Nonnull PipelineNetwork network,
                                  @Nonnull int[] contentDims,
                                  @Nullable UnaryOperator<Tensor> viewLayer,
                                  @Nullable Tensor mask,
                                  @Nonnull Tensor... styleImages) {
    assert network != null;
    this.network = network;
    //assert mask != null;
    this.mask = mask;
    if (null == contentDims) {
      throw new IllegalArgumentException();
    }
    this.contentDims = contentDims;
    this.viewLayer = viewLayer;
    this.style = styleImages;
  }

  @Nonnull
  public Tensor getMask() {
    assertAlive();
    return null == mask ? null : mask.addRef();
  }

  @Nullable
  public PipelineNetwork getNetwork() {
    assertAlive();
    return network.addRef();
  }

  public Tensor[] getStyle() {
    assertAlive();
    return RefUtil.addRef(style);
  }

  public PipelineNetwork copyNetwork() {
    assertAlive();
    return network.copyPipeline();
  }

  public void _free() {
    RefUtil.freeRef(network);
    RefUtil.freeRef(mask);
    RefUtil.freeRef(style);
    super._free();
  }

  @Nonnull
  public VisualModifierParameters withMask(@Nullable Tensor mask) {
    if (null != mask) {
      BufferedImage maskImage = mask.toRgbImage();
      RefUtil.freeRef(mask);
      mask = Tensor.fromRGB(ImageUtil.resize(maskImage, contentDims[0], contentDims[1]));
      if (null != viewLayer) {
        mask = viewLayer.apply(mask);
      }
    }
    return new VisualModifierParameters(getNetwork(), contentDims, viewLayer, mask, getStyle());
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  VisualModifierParameters addRef() {
    return (VisualModifierParameters) super.addRef();
  }

}
