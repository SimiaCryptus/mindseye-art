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

public enum VGG16 implements VisionPipelineLayer {
  VGG16_0(new int[]{0, 0}, new int[]{0, 0}, new int[]{1, 1}, new int[]{1, 1}, 3, 3, getVgg16_hdf5()::phase0),
  VGG16_1a(new int[]{0, 0}, new int[]{0, 0}, new int[]{1, 1}, new int[]{1, 1}, 3, 64, getVgg16_hdf5()::phase1a),
  VGG16_1b(new int[]{0, 0}, new int[]{0, 0}, new int[]{5, 5}, new int[]{2, 2}, 64, 128, getVgg16_hdf5()::phase1b),
  VGG16_1c(new int[]{0, 0}, new int[]{0, 0}, new int[]{7, 7}, new int[]{2, 2}, 128, 256, getVgg16_hdf5()::phase1c),
  VGG16_1d(new int[]{0, 0}, new int[]{0, 0}, new int[]{7, 7}, new int[]{2, 2}, 256, 512, getVgg16_hdf5()::phase1d),
  VGG16_1e(new int[]{0, 0}, new int[]{0, 0}, new int[]{7, 7}, new int[]{2, 2}, 512, 512, getVgg16_hdf5()::phase1e),
  VGG16_2a(new int[]{0, 0}, new int[]{0, 0}, new int[]{7, 7}, new int[]{2, 2}, 512, 512, getVgg16_hdf5()::phase2a),
  VGG16_2b(new int[]{0, 0}, new int[]{0, 0}, new int[]{7, 7}, new int[]{2, 2}, 512, 4096, getVgg16_hdf5()::phase2b),
  VGG16_3a(new int[]{0, 0}, new int[]{0, 0}, new int[]{7, 7}, new int[]{2, 2}, 512, 512, getVgg16_hdf5()::phase3a),
  VGG16_3b(new int[]{0, 0}, new int[]{0, 0}, new int[]{7, 7}, new int[]{2, 2}, 512, 512, getVgg16_hdf5()::phase3b);

  private static volatile VisionPipeline<VisionPipelineLayer> visionPipeline = null;
  private static VGG16_HDF5 vgg16_hdf5 = null;
  private final Consumer<PipelineNetwork> fn;
  private final int[] inputBorders;
  private final int[] outputBorders;
  private final int[] kenelSize;
  private final int[] strides;
  private final int inputChannels;
  private final int outputChannels;
  VGG16(int[] inputBorders, int[] outputBorders, int[] kenelSize, int[] strides, int inputChannels, int outputChannels, Consumer<PipelineNetwork> fn) {
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
        vgg16_hdf5 = (VGG16_HDF5) VGG16_HDF5.fromHDF5();
      }
    }
    return vgg16_hdf5;
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
//
//  public int[] outputDims(int... inputDims) {
//    return IntStream.range(0, inputDims.length).map(d -> {
//      int inputDim = inputDims[d];
//      if (d < 2) {
//        int stride = this.getStrides()[d];
//        return (int) Math.ceil(((double) (inputDim) / stride));
//      } else if (d == 2) {
//        if (inputDim != getInputChannels()) throw new IllegalArgumentException();
//        return getOutputChannels();
//      } else throw new IllegalArgumentException();
//    }).toArray();
//  }

  @Override
  public VisionPipeline<?> getPipeline() {
    return getVisionPipeline();
  }

}
