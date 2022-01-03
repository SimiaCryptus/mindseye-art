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
import com.simiacryptus.ref.lang.ReferenceCountingBase;

import javax.annotation.Nonnull;
import java.util.Objects;

/**
 * The type Anonymous vision pipeline layer.
 */
public class AnonymousVisionPipelineLayer extends  ReferenceCountingBase implements  VisionPipelineLayer {

  private final Layer layer;
  private final String pipelineName;
  private String name;

  /**
   * Instantiates a new Anonymous vision pipeline layer.
   *
   * @param pipelineName the pipeline name
   * @param layer        the layer
   * @param name         the name
   */
  public AnonymousVisionPipelineLayer(String pipelineName, Layer layer, String name) {
    this.layer = layer;
    this.pipelineName = pipelineName;
    this.name = name;
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    return layer.addRef();
  }

  @Nonnull
  @Override
  public PipelineNetwork getNetwork() {
    throw new RuntimeException("Not Implemented");
  }

  @Nonnull
  @Override
  public VisionPipeline getPipeline() {
    throw new RuntimeException("Not Implemented");
  }

  @Nonnull
  @Override
  public String getPipelineName() {
    return pipelineName;
  }

  @Nonnull
  @Override
  public String name() {
    return name;
  }

  @Override
  @RefIgnore
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    AnonymousVisionPipelineLayer that = (AnonymousVisionPipelineLayer) o;
    return Objects.equals(layer, that.layer) &&
        Objects.equals(pipelineName, that.pipelineName);
  }

  @Override
  @RefIgnore
  public int hashCode() {
    return Objects.hash(layer, pipelineName);
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
    layer.freeRef();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  AnonymousVisionPipelineLayer addRef() {
    return (AnonymousVisionPipelineLayer) super.addRef();
  }

  private double scale = 1.0;
  @Override
  public @Nonnull VisionPipelineLayer scale(double scale) {
    this.scale *= scale;
    return addRef();
  }

  @Override
  public @Nonnull double getScale() {
    return scale;
  }

}
