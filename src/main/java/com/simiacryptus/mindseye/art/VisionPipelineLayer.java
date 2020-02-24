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
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.wrappers.RefLinkedHashMap;
import com.simiacryptus.ref.wrappers.RefString;

import javax.annotation.Nonnull;

import static com.simiacryptus.mindseye.layers.cudnn.PoolingLayer.getPoolingLayer;

public interface VisionPipelineLayer {
  VisionPipelineLayer.Noop NOOP = new VisionPipelineLayer.Noop();

  @Nonnull
  Layer getLayer();

  @Nonnull
  default PipelineNetwork getNetwork() {
    final VisionPipeline<?> pipeline = getPipeline();
    RefLinkedHashMap<?, PipelineNetwork> layers = pipeline.getLayers();
    PipelineNetwork network = layers.get(this);
    final PipelineNetwork pipelineNetwork = network.copyPipeline();
    layers.freeRef();
    network.freeRef();
    pipeline.freeRef();
    assert pipelineNetwork != null;
    return pipelineNetwork;
  }

  @Nonnull
  VisionPipeline<?> getPipeline();

  @Nonnull
  String getPipelineName();

  @Nonnull
  String name();

  @Nonnull
  default VisionPipelineLayer prependAvgPool(int radius) {
    return prependPool(radius, PoolingLayer.PoolingMode.Avg);
  }

  @Nonnull
  default VisionPipelineLayer appendAvgPool(int radius) {
    return appendPool(radius, PoolingLayer.PoolingMode.Avg);
  }

  @Nonnull
  default VisionPipelineLayer appendMaxPool(int radius) {
    return appendPool(radius, PoolingLayer.PoolingMode.Max);
  }

  @Nonnull
  default VisionPipelineLayer prependMaxPool(int radius) {
    return prependPool(radius, PoolingLayer.PoolingMode.Max);
  }

  @Nonnull
  default VisionPipelineLayer prependPool(int radius, PoolingLayer.PoolingMode mode) {
    return prepend(getPoolingLayer(radius, mode, RefString.format("prepend(%s)", this)));
  }

  @Nonnull
  default VisionPipelineLayer appendPool(int radius, PoolingLayer.PoolingMode mode) {
    return append(getPoolingLayer(radius, mode, RefString.format("append(%s)", this)));
  }

  @Nonnull
  @RefAware
  default VisionPipelineLayer prepend(Layer layer) {
    return new PrependVisionPipelineLayer(this, layer);
  }

  @Nonnull
  @RefAware
  default VisionPipelineLayer append(Layer layer) {
    return new AppendVisionPipelineLayer(this, layer);
  }

  class Noop implements VisionPipelineLayer {

    @Nonnull
    @Override
    public Layer getLayer() {
      return new PipelineNetwork(1);
    }

    @Nonnull
    @Override
    public VisionPipeline<Noop> getPipeline() {
      return new VisionPipeline<Noop>(name(), this);
    }

    @Nonnull
    @Override
    public String getPipelineName() {
      return name();
    }

    @Nonnull
    @Override
    public String name() {
      return "NOOP";
    }
  }

}
