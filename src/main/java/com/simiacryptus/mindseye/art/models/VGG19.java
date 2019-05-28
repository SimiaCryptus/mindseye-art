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
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.network.PipelineNetwork;

import java.util.UUID;
import java.util.function.Consumer;

public enum VGG19 implements VisionPipelineLayer {
  VGG19_0(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 3, 64, getVGG19_hdf5()::phase0),
  VGG19_1a(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 3, 64, getVGG19_hdf5()::phase1a),
  VGG19_1b1(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 64, 128, getVGG19_hdf5()::phase1b1),
  VGG19_1b2(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 128, 128, getVGG19_hdf5()::phase1b2),
  VGG19_1c1(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 128, 256, getVGG19_hdf5()::phase1c1),
  VGG19_1c2(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 256, 256, getVGG19_hdf5()::phase1c2),
  VGG19_1c3(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 256, 256, getVGG19_hdf5()::phase1c3),
  VGG19_1c4(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 256, 256, getVGG19_hdf5()::phase1c4),
  VGG19_1d1(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 256, 512, getVGG19_hdf5()::phase1d1),
  VGG19_1d2(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 512, 512, getVGG19_hdf5()::phase1d2),
  VGG19_1d3(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 512, 512, getVGG19_hdf5()::phase1d3),
  VGG19_1d4(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 512, 512, getVGG19_hdf5()::phase1d4),
  VGG19_1e1(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 512, 512, getVGG19_hdf5()::phase1e1),
  VGG19_1e2(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 512, 512, getVGG19_hdf5()::phase1e2),
  VGG19_1e3(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 512, 512, getVGG19_hdf5()::phase1e3),
  VGG19_1e4(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 512, 512, getVGG19_hdf5()::phase1e4),
  VGG19_2(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 512, 4096, getVGG19_hdf5()::phase2);
//  VGG19_3a(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 3, 64, getVGG19_hdf5()::phase3a),
//  VGG19_3b(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 3, 64, getVGG19_hdf5()::phase3b);

  private static volatile VisionPipeline<VisionPipelineLayer> visionPipeline = null;
  private static VGG19_HDF5 VGG19_hdf5 = null;
  private final Consumer<PipelineNetwork> fn;
  private final int[] inputBorders;
  private final int[] outputBorders;
  private final int[] kenelSize;
  private final int[] strides;
  private final int inputChannels;
  private final int outputChannels;
  private volatile PipelineNetwork pipeline = null;

  VGG19(int[] inputBorders, int[] outputBorders, int[] kenelSize, int[] strides, int inputChannels, int outputChannels, Consumer<PipelineNetwork> fn) {
    this.fn = fn;
    this.inputChannels = inputChannels;
    this.outputChannels = outputChannels;
    this.inputBorders = inputBorders;
    this.outputBorders = outputBorders;
    this.kenelSize = kenelSize;
    this.strides = strides;
  }

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

  public static VGG19_HDF5 getVGG19_hdf5() {
    if (null == VGG19_hdf5) {
      if (null == VGG19_hdf5) {
        VGG19_hdf5 = VGG19_HDF5.fromHDF5();
      }
    }
    return VGG19_hdf5;
  }

  @Override
  public Layer getLayer() {
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
  public int[] getInputBorders() {
    return this.inputBorders;
  }

  @Override
  public int[] getOutputBorders() {
    return this.outputBorders;
  }

  @Override
  public int getInputChannels() {
    return inputChannels;
  }

  @Override
  public int getOutputChannels() {
    return outputChannels;
  }

  @Override
  public int[] getKernelSize() {
    return this.kenelSize;
  }

  @Override
  public int[] getStrides() {
    return this.strides;
  }

  @Override
  public VisionPipeline<?> getPipeline() {
    return getVisionPipeline();
  }

}
