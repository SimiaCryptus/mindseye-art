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
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.RefLinkedHashMap;
import com.simiacryptus.ref.wrappers.RefSet;
import com.simiacryptus.ref.wrappers.RefStream;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Objects;

public class PrependVisionPipelineLayer extends ReferenceCountingBase implements VisionPipelineLayer {

  private final VisionPipelineLayer inner;
  private final Layer layer;

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
  public VisionPipeline<VisionPipelineLayer> getPipeline() {
    assertAlive();
    StaticVisionPipelineLayer staticVisionPipelineLayer = new StaticVisionPipelineLayer(getPipelineName(), layer.addRef());
    final VisionPipeline<? extends VisionPipelineLayer> innerPipeline = inner.getPipeline();
    RefLinkedHashMap<? extends VisionPipelineLayer, PipelineNetwork> layers = innerPipeline.getLayers();
    RefSet<? extends VisionPipelineLayer> keySet = layers.keySet();
    layers.freeRef();
    VisionPipeline<VisionPipelineLayer> visionPipeline = new VisionPipeline<>(getPipelineName(),
        RefStream
            .concat(RefStream.of(staticVisionPipelineLayer.addRef()),
                keySet.stream().map(x -> new PrependVisionPipelineLayer(x, layer.addRef())))
            .toArray(i -> new VisionPipelineLayer[i]));
    keySet.freeRef();
    staticVisionPipelineLayer.reference.set(visionPipeline.addRef());
    staticVisionPipelineLayer.freeRef();
    innerPipeline.freeRef();
    return visionPipeline;
  }

  @Nonnull
  @Override
  public String getPipelineName() {
    return inner.getPipelineName() + "/prepend=" + layer.getName();
  }


  @Nonnull
  @Override
  public String name() {
    return inner.name() + "/prepend=" + layer.getName();
  }

  @Override
  @RefIgnore
  public boolean equals(@Nullable Object o) {
    if (this == o)
      return true;
    if (o == null || getClass() != o.getClass())
      return false;
    PrependVisionPipelineLayer that = (PrependVisionPipelineLayer) o;
    if (!Objects.equals(getPipelineName(), that.getPipelineName()))
      return false;
    if (!Objects.equals(name(), that.name()))
      return false;
    return true;
  }

  @Override
  public int hashCode() {
    return getPipelineName().hashCode() ^ name().hashCode();
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
    layer.freeRef();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  PrependVisionPipelineLayer addRef() {
    return (PrependVisionPipelineLayer) super.addRef();
  }

}
