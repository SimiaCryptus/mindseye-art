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
 * The enum Vgg 16.
 */
public enum VGG16 implements VisionPipelineLayer {
  /**
   * Vgg 16 0 a vgg 16.
   */
  VGG16_0a(x -> {
    x.freeRef();
  }),
  /**
   * Vgg 16 0 b vgg 16.
   */
  VGG16_0b(pipeline -> {
    getVgg16_hdf5().phase0b(pipeline);
  }),
  /**
   * Vgg 16 1 a vgg 16.
   */
  VGG16_1a(pipeline -> {
    getVgg16_hdf5().phase1a(pipeline);
  }),
  /**
   * Vgg 16 1 b 1 vgg 16.
   */
  VGG16_1b1(pipeline -> {
    getVgg16_hdf5().phase1b1(pipeline);
  }),
  /**
   * Vgg 16 1 b 2 vgg 16.
   */
  VGG16_1b2(pipeline -> {
    getVgg16_hdf5().phase1b2(pipeline);
  }),
  /**
   * Vgg 16 1 c 1 vgg 16.
   */
  VGG16_1c1(pipeline -> {
    getVgg16_hdf5().phase1c1(pipeline);
  }),
  /**
   * Vgg 16 1 c 2 vgg 16.
   */
  VGG16_1c2(pipeline -> {
    getVgg16_hdf5().phase1c2(pipeline);
  }),
  /**
   * Vgg 16 1 c 3 vgg 16.
   */
  VGG16_1c3(pipeline -> {
    getVgg16_hdf5().phase1c3(pipeline);
  }),
  /**
   * Vgg 16 1 d 1 vgg 16.
   */
  VGG16_1d1(pipeline -> {
    getVgg16_hdf5().phase1d1(pipeline);
  }),
  /**
   * Vgg 16 1 d 2 vgg 16.
   */
  VGG16_1d2(pipeline -> {
    getVgg16_hdf5().phase1d2(pipeline);
  }),
  /**
   * Vgg 16 1 d 3 vgg 16.
   */
  VGG16_1d3(pipeline -> {
    getVgg16_hdf5().phase1d3(pipeline);
  }),
  /**
   * Vgg 16 1 e 1 vgg 16.
   */
  VGG16_1e1(pipeline -> {
    getVgg16_hdf5().phase1e1(pipeline);
  }),
  /**
   * Vgg 16 1 e 2 vgg 16.
   */
  VGG16_1e2(pipeline -> {
    getVgg16_hdf5().phase1e2(pipeline);
  }),
  /**
   * Vgg 16 1 e 3 vgg 16.
   */
  VGG16_1e3(pipeline -> {
    getVgg16_hdf5().phase1e3(pipeline);
  }),
  /**
   * Vgg 16 2 vgg 16.
   */
  VGG16_2(pipeline -> {
    getVgg16_hdf5().phase2(pipeline);
  }),
  /**
   * Vgg 16 3 a vgg 16.
   */
  VGG16_3a(pipeline -> {
    getVgg16_hdf5().phase3a(pipeline);
  }),
  /**
   * Vgg 16 3 b vgg 16.
   */
  VGG16_3b(pipeline -> {
    getVgg16_hdf5().phase3b(pipeline);
  });

  @Nullable
  private static volatile VisionPipeline visionPipeline = null;
  @Nullable
  private static VGG16_HDF5 vgg16_hdf5 = null;
  private final RefConsumer<PipelineNetwork> fn;

  VGG16(RefConsumer<PipelineNetwork> fn) {
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
   * Gets vgg 16 hdf 5.
   *
   * @return the vgg 16 hdf 5
   */
  @Nonnull
  public static VGG16_HDF5 getVgg16_hdf5() {
    if (null == vgg16_hdf5) {
      vgg16_hdf5 = VGG16_HDF5.fromHDF5();
    }
    return vgg16_hdf5;
  }

  /**
   * Gets vision pipeline.
   *
   * @return the vision pipeline
   */
  @Nullable
  public static VisionPipeline getVisionPipeline() {
    if (null == visionPipeline) {
      synchronized (VGG16.class) {
        if (null == visionPipeline) {
          visionPipeline = new VisionPipeline(VGG16.class.getSimpleName(), VGG16.values());
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
