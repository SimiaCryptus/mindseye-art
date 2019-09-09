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
import com.simiacryptus.mindseye.network.PipelineNetwork;

import java.util.UUID;
import java.util.function.Consumer;

public enum VGG16 implements VisionPipelineLayer {
  VGG16_0a(x -> {
  }),
  VGG16_0b(getVgg16_hdf5()::phase0b),
  VGG16_1a(getVgg16_hdf5()::phase1a),
  VGG16_1b1(getVgg16_hdf5()::phase1b1),
  VGG16_1b2(getVgg16_hdf5()::phase1b2),
  VGG16_1c1(getVgg16_hdf5()::phase1c1),
  VGG16_1c2(getVgg16_hdf5()::phase1c2),
  VGG16_1c3(getVgg16_hdf5()::phase1c3),
  VGG16_1d1(getVgg16_hdf5()::phase1d1),
  VGG16_1d2(getVgg16_hdf5()::phase1d2),
  VGG16_1d3(getVgg16_hdf5()::phase1d3),
  VGG16_1e1(getVgg16_hdf5()::phase1e1),
  VGG16_1e2(getVgg16_hdf5()::phase1e2),
  VGG16_1e3(getVgg16_hdf5()::phase1e3),
  VGG16_2(getVgg16_hdf5()::phase2),
  VGG16_3a(getVgg16_hdf5()::phase3a),
  VGG16_3b(getVgg16_hdf5()::phase3b);

  private static volatile VisionPipeline<VisionPipelineLayer> visionPipeline = null;
  private static VGG16_HDF5 vgg16_hdf5 = null;
  private final Consumer<PipelineNetwork> fn;
  private volatile PipelineNetwork pipeline = null;

  VGG16(Consumer<PipelineNetwork> fn) {
    this.fn = fn;
  }

  public static VisionPipeline<VisionPipelineLayer> getVisionPipeline() {
    if (null == visionPipeline) {
      synchronized (VGG16.class) {
        if (null == visionPipeline) {
          visionPipeline = new VisionPipeline<>(VGG16.class.getSimpleName(), VGG16.values());
        }
      }
    }
    return visionPipeline;
  }

  public static VGG16_HDF5 getVgg16_hdf5() {
    if (null == vgg16_hdf5) {
      if (null == vgg16_hdf5) {
        vgg16_hdf5 = VGG16_HDF5.fromHDF5();
      }
    }
    return vgg16_hdf5;
  }

  @Override
  public PipelineNetwork getLayer() {
    if (null == pipeline) {
      synchronized (this) {
        if (null == pipeline) {
          pipeline = new PipelineNetwork(1, UUID.nameUUIDFromBytes(name().getBytes()), name());
          fn.accept(pipeline);
        }
      }
    }
    return pipeline.copyPipeline();
  }

  @Override
  public String getPipelineName() {
    return getVisionPipeline().name;
  }

  @Override
  public VisionPipeline<?> getPipeline() {
    return getVisionPipeline();
  }

}
