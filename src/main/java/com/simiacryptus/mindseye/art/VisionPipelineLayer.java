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
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;

import static com.simiacryptus.mindseye.layers.cudnn.PoolingLayer.getPoolingLayer;

public @com.simiacryptus.ref.lang.RefAware
interface VisionPipelineLayer {
  final VisionPipelineLayer.Noop NOOP = new VisionPipelineLayer.Noop();

  @Nonnull
  Layer getLayer();

  @Nonnull
  default PipelineNetwork getNetwork() {
    final VisionPipeline<?> pipeline = getPipeline();
    final PipelineNetwork pipelineNetwork = pipeline.getLayers().get(this).copyPipeline();
    pipeline.freeRef();
    return pipelineNetwork;
  }

  @Nonnull
  VisionPipeline<?> getPipeline();

  @Nonnull
  String getPipelineName();

  @Nonnull
  String name();

  default VisionPipelineLayer prependAvgPool(int radius) {
    return prependPool(radius, PoolingLayer.PoolingMode.Avg);
  }

  default VisionPipelineLayer appendAvgPool(int radius) {
    return appendPool(radius, PoolingLayer.PoolingMode.Avg);
  }

  default VisionPipelineLayer appendMaxPool(int radius) {
    return appendPool(radius, PoolingLayer.PoolingMode.Max);
  }

  default VisionPipelineLayer prependMaxPool(int radius) {
    return prependPool(radius, PoolingLayer.PoolingMode.Max);
  }

  @NotNull
  default VisionPipelineLayer prependPool(int radius, PoolingLayer.PoolingMode mode) {
    return prepend(getPoolingLayer(radius, mode, String.format("prepend(%s)", this)));
  }

  @NotNull
  default VisionPipelineLayer appendPool(int radius, PoolingLayer.PoolingMode mode) {
    return append(getPoolingLayer(radius, mode, String.format("append(%s)", this)));
  }

  @NotNull
  default VisionPipelineLayer prepend(Layer layer) {
    return new PrependVisionPipelineLayer(this, layer);
  }

  @NotNull
  default VisionPipelineLayer append(Layer layer) {
    return new AppendVisionPipelineLayer(this, layer);
  }

  public static @com.simiacryptus.ref.lang.RefAware
  class Noop implements VisionPipelineLayer {

    @Nonnull
    @Override
    public Layer getLayer() {
      return new PipelineNetwork(1);
    }

    @Nonnull
    @Override
    public VisionPipeline<Noop> getPipeline() {
      return new VisionPipeline<Noop>(name(), this) {

        public @SuppressWarnings("unused")
        void _free() {
        }
      };
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
