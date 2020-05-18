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

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;

/**
 * The type Prepend vision pipeline layer.
 */
public class PrependVisionPipelineLayer extends ReferenceCountingBase implements VisionPipelineLayer {

  private final VisionPipelineLayer inner;
  private final Layer layer;

  /**
   * Instantiates a new Prepend vision pipeline layer.
   *
   * @param inner the inner
   * @param layer the layer
   */
  public PrependVisionPipelineLayer(VisionPipelineLayer inner, Layer layer) {
    this.inner = inner;
    this.layer = layer;
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    return inner.getLayer();
  }

  @Nonnull
  @Override
  public VisionPipeline getPipeline() {
    assertAlive();
    final VisionPipeline innerPipeline = inner.getPipeline();
    RefList<VisionPipelineLayer> innerLayers = innerPipeline.getLayerList();
    innerPipeline.freeRef();
    innerLayers.add(0, new AnonymousVisionPipelineLayer(getPipelineName(), layer.addRef(), "prepend:" + layer.getName()));
    return new VisionPipeline(getPipelineName(), innerLayers);
  }

  @Nonnull
  @Override
  public String getPipelineName() {
    return inner.getPipelineName() + "/prepend=" + layer.getName();
  }


  @Nonnull
  @Override
  public String name() {
    return inner.name();
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
    layer.freeRef();
    inner.freeRef();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  PrependVisionPipelineLayer addRef() {
    return (PrependVisionPipelineLayer) super.addRef();
  }

}
