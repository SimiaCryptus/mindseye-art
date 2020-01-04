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

import java.util.Objects;

public @com.simiacryptus.ref.lang.RefAware
class PrependVisionPipelineLayer extends ReferenceCountingBase
    implements VisionPipelineLayer {

  private final VisionPipelineLayer inner;
  private final Layer layer;

  public PrependVisionPipelineLayer(VisionPipelineLayer inner, Layer layer) {
    this.inner = inner;
    this.layer = layer;
  }

  @Override
  public Layer getLayer() {
    return inner.getLayer();
  }

  @Override
  public VisionPipeline<VisionPipelineLayer> getPipeline() {
    StaticVisionPipelineLayer staticVisionPipelineLayer = new StaticVisionPipelineLayer(getPipelineName(), layer);
    final VisionPipeline<?> innerPipeline = inner.getPipeline();
    VisionPipeline<VisionPipelineLayer> visionPipeline = new VisionPipeline<>(getPipelineName(),
        com.simiacryptus.ref.wrappers.RefStream
            .concat(com.simiacryptus.ref.wrappers.RefStream.of(staticVisionPipelineLayer),
                innerPipeline.getLayers().keySet().stream()
                    .map(x -> new com.simiacryptus.mindseye.art.PrependVisionPipelineLayer(x, layer)))
            .toArray(i -> new VisionPipelineLayer[i]));
    staticVisionPipelineLayer.reference.set(visionPipeline);
    innerPipeline.freeRef();
    return visionPipeline;
  }

  @Override
  public String getPipelineName() {
    return inner.getPipelineName() + "/prepend=" + layer.getName();
  }

  public static @SuppressWarnings("unused")
  PrependVisionPipelineLayer[] addRefs(PrependVisionPipelineLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(PrependVisionPipelineLayer::addRef)
        .toArray((x) -> new PrependVisionPipelineLayer[x]);
  }

  public static @SuppressWarnings("unused")
  PrependVisionPipelineLayer[][] addRefs(
      PrependVisionPipelineLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(PrependVisionPipelineLayer::addRefs)
        .toArray((x) -> new PrependVisionPipelineLayer[x][]);
  }

  @Override
  public String name() {
    return inner.name() + "/prepend=" + layer.getName();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o)
      return true;
    if (o == null || getClass() != o.getClass())
      return false;
    com.simiacryptus.mindseye.art.PrependVisionPipelineLayer that = (com.simiacryptus.mindseye.art.PrependVisionPipelineLayer) o;
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
  }

  public @Override
  @SuppressWarnings("unused")
  PrependVisionPipelineLayer addRef() {
    return (PrependVisionPipelineLayer) super.addRef();
  }

}
