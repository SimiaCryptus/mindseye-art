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
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefConsumer;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.UUID;

public enum VGG19 implements VisionPipelineLayer, ReferenceCounting {
  VGG19_0a(p -> {
  }), VGG19_0b(getVGG19_hdf5()::phase0), VGG19_1a(getVGG19_hdf5()::phase1a), VGG19_1b1(getVGG19_hdf5()::phase1b1),
  VGG19_1b2(getVGG19_hdf5()::phase1b2), VGG19_1c1(getVGG19_hdf5()::phase1c1), VGG19_1c2(getVGG19_hdf5()::phase1c2),
  VGG19_1c3(getVGG19_hdf5()::phase1c3), VGG19_1c4(getVGG19_hdf5()::phase1c4), VGG19_1d1(getVGG19_hdf5()::phase1d1),
  VGG19_1d2(getVGG19_hdf5()::phase1d2), VGG19_1d3(getVGG19_hdf5()::phase1d3), VGG19_1d4(getVGG19_hdf5()::phase1d4),
  VGG19_1e1(getVGG19_hdf5()::phase1e1), VGG19_1e2(getVGG19_hdf5()::phase1e2), VGG19_1e3(getVGG19_hdf5()::phase1e3),
  VGG19_1e4(getVGG19_hdf5()::phase1e4), VGG19_2(getVGG19_hdf5()::phase2);
  //  VGG19_3a(getVGG19_hdf5()::phase3a),
  //  VGG19_3b(getVGG19_hdf5()::phase3b);

  @Nullable
  private static volatile VisionPipeline<VisionPipelineLayer> visionPipeline = null;
  @Nullable
  private static VGG19_HDF5 VGG19_hdf5 = null;
  private final RefConsumer<PipelineNetwork> fn;

  VGG19(RefConsumer<PipelineNetwork> fn) {
    this.fn = fn;
  }

  @Nonnull
  @Override
  public PipelineNetwork getLayer() {
    PipelineNetwork pipeline = new PipelineNetwork(1, UUID.nameUUIDFromBytes(name().getBytes()), name());
    fn.accept(pipeline);
    return pipeline.copyPipeline();
  }

  @Nonnull
  @Override
  public VisionPipeline<?> getPipeline() {
    return getVisionPipeline().addRef();
  }

  @Nonnull
  @Override
  public String getPipelineName() {
    return getVisionPipeline().name;
  }

  @Nonnull
  public static VGG19_HDF5 getVGG19_hdf5() {
    if (null == VGG19_hdf5) {
      VGG19_hdf5 = VGG19_HDF5.fromHDF5();
    }
    return VGG19_hdf5;
  }

  @Nullable
  public static VisionPipeline<VisionPipelineLayer> getVisionPipeline() {
    if (null == visionPipeline) {
      synchronized (VGG19.class) {
        if (null == visionPipeline) {
          visionPipeline = new VisionPipeline<>(VGG19.class.getSimpleName(), VGG19.values());
        }
      }
    }
    return visionPipeline;
  }

}
