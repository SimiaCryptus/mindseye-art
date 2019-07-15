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

import java.util.Objects;

public class AppendVisionPipelineLayer implements VisionPipelineLayer {

  private final VisionPipelineLayer inner;
  private final Layer layer;

  public AppendVisionPipelineLayer(VisionPipelineLayer inner, Layer layer) {
    this.inner = inner;
    this.layer = layer;
  }

  @Override
  public String name() {
    return inner.name();
  }

  @Override
  public VisionPipeline<VisionPipelineLayer> getPipeline() {
    return new VisionPipeline<>(
        getPipelineName(),
        inner.getPipeline().getLayers().stream().map(x -> new com.simiacryptus.mindseye.art.AppendVisionPipelineLayer(x, layer)).toArray(i -> new VisionPipelineLayer[i])
    );
  }

  @Override
  public String getPipelineName() {
    return inner.getPipelineName() + "/append=" + layer.getName();
  }

  @Override
  public Layer getLayer() {
    return inner.getLayer();
  }

  @Override
  public PipelineNetwork getNetwork() {
    return PipelineNetwork.sequence(
        inner.getNetwork(),
        PipelineNetwork.build(1, layer)
    );
  }

  @Override
  public int[] getInputBorders() {
    return inner.getInputBorders();
  }

  @Override
  public int[] getOutputBorders() {
    return inner.getOutputBorders();
  }

  @Override
  public int getInputChannels() {
    return inner.getInputChannels();
  }

  @Override
  public int getOutputChannels() {
    return inner.getOutputChannels();
  }

  @Override
  public int[] getKernelSize() {
    return inner.getKernelSize();
  }

  @Override
  public int[] getStrides() {
    return inner.getStrides();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    com.simiacryptus.mindseye.art.AppendVisionPipelineLayer that = (com.simiacryptus.mindseye.art.AppendVisionPipelineLayer) o;
    if (!Objects.equals(getPipelineName(), that.getPipelineName())) return false;
    if (!Objects.equals(name(), that.name())) return false;
    return true;
  }

  @Override
  public int hashCode() {
    return getPipelineName().hashCode() ^ name().hashCode();
  }
}
