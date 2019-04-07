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

import java.util.function.Consumer;
import java.util.stream.IntStream;

public enum VGG19 implements VisionPipelineLayer {
  VGG19_0(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 3, 64, getVGG19_hdf5()::phase0),
  VGG19_1a(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 3, 64, getVGG19_hdf5()::phase1a),
  VGG19_1b(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 3, 64, getVGG19_hdf5()::phase1b),
  VGG19_1c(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 3, 64, getVGG19_hdf5()::phase1c),
  VGG19_1d(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 3, 64, getVGG19_hdf5()::phase1d),
  VGG19_1e(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 3, 64, getVGG19_hdf5()::phase1e),
  VGG19_2a(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 3, 64, getVGG19_hdf5()::phase2a),
  VGG19_2b(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 3, 64, getVGG19_hdf5()::phase2b),
  VGG19_3a(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 3, 64, getVGG19_hdf5()::phase3a),
  VGG19_3b(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 3, 64, getVGG19_hdf5()::phase3b);

  private static volatile VisionPipeline<VisionPipelineLayer> visionPipeline = null;
  private static VGG19_HDF5 VGG19_hdf5 = null;
  private final Consumer<PipelineNetwork> fn;
  private final int[] inputBorders;
  private final int[] outputBorders;
  private final int[] kenelSize;
  private final int[] strides;
  private final int inputChannels;
  private final int outputChannels;

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
        VGG19_hdf5 = (VGG19_HDF5) VGG19_HDF5.fromHDF5();
      }
    }
    return VGG19_hdf5;
  }

  @Override
  public Layer getLayer() {
    PipelineNetwork pipeline = new PipelineNetwork();
    fn.accept(pipeline);
    return pipeline;
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

  public int[] outputDims(int... inputDims) {
    return IntStream.range(0, inputDims.length).map(d -> {
      int inputDim = inputDims[d];
      if (d < 2) {
        int stride = this.getStrides()[d];
        return (int) Math.ceil(((double) (inputDim) / stride));
      } else if (d == 2) {
        if (inputDim != getInputChannels()) throw new IllegalArgumentException();
        return getOutputChannels();
      } else throw new IllegalArgumentException();
    }).toArray();
  }

  @Override
  public VisionPipeline<?> getPipeline() {
    return getVisionPipeline();
  }

}
