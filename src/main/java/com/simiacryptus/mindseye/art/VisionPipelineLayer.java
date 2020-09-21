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
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.LoggingLayer;
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer;
import com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefString;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;

import java.util.concurrent.atomic.AtomicReference;

import static com.simiacryptus.mindseye.layers.cudnn.PoolingLayer.getPoolingLayer;

/**
 * The interface Vision pipeline layer.
 */
public interface VisionPipelineLayer extends ReferenceCounting {
  /**
   * The constant NOOP.
   */
  public static final VisionPipelineLayer.Noop NOOP = new VisionPipelineLayer.Noop();

  /**
   * Gets layer.
   *
   * @return the layer
   */
  @Nonnull
  public abstract Layer getLayer();

  /**
   * Gets network.
   *
   * @return the network
   */
  @Nonnull
  default PipelineNetwork getNetwork() {
    final VisionPipeline pipeline = getPipeline();
    final PipelineNetwork network;
    try {
      network = pipeline.get(name());
    } finally {
      pipeline.freeRef();
    }
    return PipelineNetwork.getCopy(network);
  }

  /**
   * Gets pipeline.
   *
   * @return the pipeline
   */
  @Nonnull
  public abstract VisionPipeline getPipeline();

  /**
   * Gets pipeline name.
   *
   * @return the pipeline name
   */
  @Nonnull
  public abstract String getPipelineName();

  /**
   * Name string.
   *
   * @return the string
   */
  @Nonnull
  public abstract String name();

  /**
   * Prepend avg pool vision pipeline layer.
   *
   * @param radius the radius
   * @return the vision pipeline layer
   */
  @Nonnull
  default VisionPipelineLayer prependAvgPool(int radius) {
    return prependPool(radius, PoolingLayer.PoolingMode.Avg);
  }

  /**
   * Append avg pool vision pipeline layer.
   *
   * @param radius the radius
   * @return the vision pipeline layer
   */
  @Nonnull
  default VisionPipelineLayer appendAvgPool(int radius) {
    return appendPool(radius, PoolingLayer.PoolingMode.Avg);
  }

  /**
   * Append max pool vision pipeline layer.
   *
   * @param radius the radius
   * @return the vision pipeline layer
   */
  @Nonnull
  default  VisionPipelineLayer appendMaxPool(int radius) {
    return appendPool(radius, PoolingLayer.PoolingMode.Max);
  }

  /**
   * Prepend max pool vision pipeline layer.
   *
   * @param radius the radius
   * @return the vision pipeline layer
   */
  @Nonnull
  default  VisionPipelineLayer prependMaxPool(int radius) {
    return prependPool(radius, PoolingLayer.PoolingMode.Max);
  }

  /**
   * Prepend pool vision pipeline layer.
   *
   * @param radius the radius
   * @param mode   the mode
   * @return the vision pipeline layer
   */
  @Nonnull
  default  VisionPipelineLayer prependPool(int radius, PoolingLayer.PoolingMode mode) {
    return prepend(getPoolingLayer(radius, mode, RefString.format("prepend(%s)", this.addRef())));
  }

  /**
   * Append pool vision pipeline layer.
   *
   * @param radius the radius
   * @param mode   the mode
   * @return the vision pipeline layer
   */
  @Nonnull
  default  VisionPipelineLayer appendPool(int radius, PoolingLayer.PoolingMode mode) {
    return append(getPoolingLayer(radius, mode, RefString.format("append(%s)", this.addRef())));
  }

  /**
   * Prepend vision pipeline layer.
   *
   * @param layer the layer
   * @return the vision pipeline layer
   */
  @Nonnull
  @RefAware
  default  VisionPipelineLayer prepend(Layer layer) {
    return new PrependVisionPipelineLayer(this.addRef(), layer);
  }

  @Override
  default  VisionPipelineLayer addRef() {
    return this;
  }

  /**
   * Append vision pipeline layer.
   *
   * @param layer the layer
   * @return the vision pipeline layer
   */
  @Nonnull
  @RefAware
  default  VisionPipelineLayer append(Layer layer) {
    return new AppendVisionPipelineLayer(this.addRef(), layer);
  }

  /**
   * With logging vision pipeline layer.
   *
   * @return the vision pipeline layer
   */
  @Nonnull
  @RefAware
  default  VisionPipelineLayer withLogging() {
    LoggingLayer loggingLayer = new LoggingLayer(LoggingLayer.DetailLevel.Statistics);
    loggingLayer.setName(name());
    loggingLayer.setLogFeedback(false);
    return new AppendVisionPipelineLayer<>(this.addRef(), loggingLayer);
  }

  /**
   * The type Noop.
   */
  public static class Noop implements VisionPipelineLayer {


    @Nonnull
    @Override
    public Layer getLayer() {
      return new PipelineNetwork(1);
    }

    @Nonnull
    @Override
    public VisionPipeline getPipeline() {
      return new VisionPipeline(name(), this.addRef());
    }

    @Nonnull
    @Override
    public String getPipelineName() {
      return name();
    }

    @Override
    public Noop addRef() {
      return this;
    }

    @Nonnull
    @Override
    public String name() {
      return "NOOP";
    }
  }

}
