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
import com.simiacryptus.ref.lang.ReferenceCountingBase;

import javax.annotation.Nonnull;

/**
 * The type Append vision pipeline layer.
 *
 * @param <T> the type parameter
 */
public class AppendVisionPipelineLayer<T extends VisionPipelineLayer> extends  ReferenceCountingBase implements  VisionPipelineLayer {

  private final T inner;
  private final Layer layer;

  /**
   * Instantiates a new Append vision pipeline layer.
   *
   * @param inner the inner
   * @param layer the layer
   */
  public AppendVisionPipelineLayer(T inner, Layer layer) {
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
  public PipelineNetwork getNetwork() {
    assertAlive();
    final PipelineNetwork network = inner.getNetwork();
    network.add(layer.addRef()).freeRef();
    return network;
  }

  @Nonnull
  @Override
  public VisionPipeline getPipeline() {
    assertAlive();
    return inner.getPipeline();
  }

  @Nonnull
  @Override
  public String getPipelineName() {
    return inner.getPipelineName() + "/append=" + layer.getName();
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
  AppendVisionPipelineLayer addRef() {
    return (AppendVisionPipelineLayer) super.addRef();
  }
}
