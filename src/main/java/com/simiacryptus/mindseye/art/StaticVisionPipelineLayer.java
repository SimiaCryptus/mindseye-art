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

import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

public class StaticVisionPipelineLayer implements VisionPipelineLayer {
  public final AtomicReference<VisionPipeline<?>> reference = new AtomicReference<>();

  private final Layer layer;
  private final String pipelineName;

  public StaticVisionPipelineLayer(String pipelineName, Layer layer) {
    this.layer = layer;
    this.pipelineName = pipelineName;
  }

  @Override
  public String name() {
    return layer.getName();
  }

  @Override
  public VisionPipeline<?> getPipeline() {
    return reference.get();
  }

  @Override
  public String getPipelineName() {
    return pipelineName;
  }

  @Override
  public Layer getLayer() {
    return layer.addRef();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    com.simiacryptus.mindseye.art.StaticVisionPipelineLayer that = (com.simiacryptus.mindseye.art.StaticVisionPipelineLayer) o;
    if (!Objects.equals(getPipelineName(), that.getPipelineName())) return false;
    if (!Objects.equals(name(), that.name())) return false;
    return true;
  }

  @Override
  public int hashCode() {
    return getPipelineName().hashCode() ^ name().hashCode();
  }

}
