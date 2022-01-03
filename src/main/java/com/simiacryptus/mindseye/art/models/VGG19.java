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
import com.simiacryptus.ref.wrappers.RefConsumer;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.UUID;

/**
 * The enum Vgg 19.
 */
public enum VGG19 implements VisionPipelineLayer {
  /**
   * Vgg 19 0 a vgg 19.
   */
  VGG19_0a(p -> {
    p.freeRef();
  }),
  /**
   * Vgg 19 0 b vgg 19.
   */
  VGG19_0b(pipeline -> {
    getVGG19_hdf5().phase0(pipeline);
  }),
  /**
   * Vgg 19 1 a vgg 19.
   */
  VGG19_1a(pipeline -> {
    getVGG19_hdf5().phase1a(pipeline);
  }),
  /**
   * Vgg 19 1 b 1 vgg 19.
   */
  VGG19_1b1(pipeline -> {
    getVGG19_hdf5().phase1b1(pipeline);
  }),
  /**
   * Vgg 19 1 b 2 vgg 19.
   */
  VGG19_1b2(pipeline -> {
    getVGG19_hdf5().phase1b2(pipeline);
  }),
  /**
   * Vgg 19 1 c 1 vgg 19.
   */
  VGG19_1c1(pipeline -> {
    getVGG19_hdf5().phase1c1(pipeline);
  }),
  /**
   * Vgg 19 1 c 2 vgg 19.
   */
  VGG19_1c2(pipeline -> {
    getVGG19_hdf5().phase1c2(pipeline);
  }),
  /**
   * Vgg 19 1 c 3 vgg 19.
   */
  VGG19_1c3(pipeline -> {
    getVGG19_hdf5().phase1c3(pipeline);
  }),
  /**
   * Vgg 19 1 c 4 vgg 19.
   */
  VGG19_1c4(pipeline -> {
    getVGG19_hdf5().phase1c4(pipeline);
  }),
  /**
   * Vgg 19 1 d 1 vgg 19.
   */
  VGG19_1d1(pipeline -> {
    getVGG19_hdf5().phase1d1(pipeline);
  }),
  /**
   * Vgg 19 1 d 2 vgg 19.
   */
  VGG19_1d2(pipeline -> {
    getVGG19_hdf5().phase1d2(pipeline);
  }),
  /**
   * Vgg 19 1 d 3 vgg 19.
   */
  VGG19_1d3(pipeline -> {
    getVGG19_hdf5().phase1d3(pipeline);
  }),
  /**
   * Vgg 19 1 d 4 vgg 19.
   */
  VGG19_1d4(pipeline -> {
    getVGG19_hdf5().phase1d4(pipeline);
  }),
  /**
   * Vgg 19 1 e 1 vgg 19.
   */
  VGG19_1e1(pipeline -> {
    getVGG19_hdf5().phase1e1(pipeline);
  }),
  /**
   * Vgg 19 1 e 2 vgg 19.
   */
  VGG19_1e2(pipeline -> {
    getVGG19_hdf5().phase1e2(pipeline);
  }),
  /**
   * Vgg 19 1 e 3 vgg 19.
   */
  VGG19_1e3(pipeline -> {
    getVGG19_hdf5().phase1e3(pipeline);
  }),
  /**
   * Vgg 19 1 e 4 vgg 19.
   */
  VGG19_1e4(pipeline -> {
    getVGG19_hdf5().phase1e4(pipeline);
  }),
  /**
   * Vgg 19 2 vgg 19.
   */
  VGG19_2(pipeline -> {
    getVGG19_hdf5().phase2(pipeline);
  });
  //  VGG19_3a(getVGG19_hdf5()::phase3a),
  //  VGG19_3b(getVGG19_hdf5()::phase3b);

  @Nullable
  private static volatile VisionPipeline visionPipeline = null;
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
    fn.accept(pipeline.addRef());
    PipelineNetwork copyPipeline = pipeline.copyPipeline();
    pipeline.freeRef();
    return copyPipeline;
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
   * Gets vgg 19 hdf 5.
   *
   * @return the vgg 19 hdf 5
   */
  @Nonnull
  public static VGG19_HDF5 getVGG19_hdf5() {
    if (null == VGG19_hdf5) {
      VGG19_hdf5 = VGG19_HDF5.fromHDF5();
    }
    return VGG19_hdf5;
  }

  /**
   * Gets vision pipeline.
   *
   * @return the vision pipeline
   */
  @Nullable
  public static VisionPipeline getVisionPipeline() {
    if (null == visionPipeline) {
      synchronized (VGG19.class) {
        if (null == visionPipeline) {
          visionPipeline = new VisionPipeline(VGG19.class.getSimpleName(), VGG19.values());
        }
      }
    }
    return visionPipeline.addRef();
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
