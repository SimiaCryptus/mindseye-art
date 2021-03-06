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

package com.simiacryptus.mindseye.art.models;

import com.simiacryptus.mindseye.art.VisionPipeline;
import com.simiacryptus.mindseye.art.VisionPipelineLayer;
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.wrappers.RefConsumer;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.UUID;

/**
 * The enum Pooling pipeline.
 */
public enum PoolingPipeline implements VisionPipelineLayer {
  /**
   * The Pooling 0.
   */
  Pooling0(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 3, 3,
      (PipelineNetwork pipeline) -> {
        pipeline.freeRef();
      }),
  /**
   * The Pooling 2.
   */
  Pooling2(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 3, 3),
  /**
   * The Pooling 4.
   */
  Pooling4(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 3, 3),
  /**
   * The Pooling 8.
   */
  Pooling8(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 3, 3),
  /**
   * The Pooling 16.
   */
  Pooling16(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 3, 3),
  /**
   * The Pooling 32.
   */
  Pooling32(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 3, 3),
  /**
   * The Pooling 64.
   */
  Pooling64(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 3, 3),
  /**
   * The Pooling 128.
   */
  Pooling128(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 3, 3);

  @Nullable
  private static volatile VisionPipeline visionPipeline = null;
  private final RefConsumer<PipelineNetwork> fn;
  private final int[] inputBorders;
  private final int[] outputBorders;
  private final int[] kenelSize;
  private final int[] strides;
  private final int inputChannels;
  private final int outputChannels;

  PoolingPipeline(int[] inputBorders, int[] outputBorders, int[] kenelSize, int[] strides, int inputChannels,
                  int outputChannels) {
    this(inputBorders, outputBorders, kenelSize, strides, inputChannels, outputChannels,
        (PipelineNetwork pipeline) -> {
          PoolingLayer poolingLayer = new PoolingLayer();
          poolingLayer.setStrideXY(2, 2);
          poolingLayer.setWindowXY(2, 2);
          poolingLayer.setMode(PoolingLayer.PoolingMode.Avg);
          pipeline.add(poolingLayer).freeRef();
          pipeline.freeRef();
        });
  }

  PoolingPipeline(int[] inputBorders, int[] outputBorders, int[] kenelSize, int[] strides, int inputChannels,
                  int outputChannels, RefConsumer<PipelineNetwork> fn) {
    this.fn = fn;
    this.inputChannels = inputChannels;
    this.outputChannels = outputChannels;
    this.inputBorders = inputBorders;
    this.outputBorders = outputBorders;
    this.kenelSize = kenelSize;
    this.strides = strides;
  }

  @Nonnull
  @Override
  public PipelineNetwork getLayer() {
    PipelineNetwork layer = new PipelineNetwork(1, UUID.nameUUIDFromBytes(name().getBytes()), name());
    fn.accept(layer.addRef());
    return layer;
  }

  @Nonnull
  @Override
  public VisionPipeline getPipeline() {
    return getVisionPipeline();
  }

  @Nonnull
  @Override
  public String getPipelineName() {
    VisionPipeline visionPipeline = getVisionPipeline();
    String name = visionPipeline.name;
    visionPipeline.freeRef();
    return name;
  }

  /**
   * Gets vision pipeline.
   *
   * @return the vision pipeline
   */
  @Nullable
  public static VisionPipeline getVisionPipeline() {
    if (null == visionPipeline) {
      synchronized (PoolingPipeline.class) {
        if (null == visionPipeline) {
          visionPipeline = new VisionPipeline(PoolingPipeline.class.getSimpleName(), PoolingPipeline.values());
        }
      }
    }
    return visionPipeline.addRef();
  }

}
