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
import com.simiacryptus.mindseye.art.util.ImageArtUtil;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.tensorflow.ImageNetworkPipeline;

import java.util.Map;

public enum Inception5H implements VisionPipelineLayer {
  Inc5H_1a(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 3, 64, "conv2d0"),
  Inc5H_2a(new int[]{2, 2}, new int[]{4, 4}, new int[]{7, 7}, new int[]{2, 2}, 64, 192, "localresponsenorm1"),
  Inc5H_3a(new int[]{0, 0}, new int[]{1, 1}, new int[]{3, 3}, new int[]{2, 2}, 192, 256, "mixed3a"),
  Inc5H_3b(new int[]{0, 0}, new int[]{0, 0}, new int[]{1, 1}, new int[]{1, 1}, 256, 480, "mixed3b"),
  Inc5H_4a(new int[]{0, 0}, new int[]{1, 1}, new int[]{3, 3}, new int[]{2, 2}, 480, 508, "mixed4a"),
  Inc5H_4b(new int[]{0, 0}, new int[]{0, 0}, new int[]{1, 1}, new int[]{1, 1}, 508, 512, "mixed4b"),
  Inc5H_4c(new int[]{0, 0}, new int[]{0, 0}, new int[]{1, 1}, new int[]{1, 1}, 512, 512, "mixed4c"),
  Inc5H_4d(new int[]{0, 0}, new int[]{0, 0}, new int[]{1, 1}, new int[]{1, 1}, 512, 528, "mixed4d"),
  Inc5H_4e(new int[]{0, 0}, new int[]{0, 0}, new int[]{1, 1}, new int[]{1, 1}, 528, 832, "mixed4e"),
  Inc5H_5a(new int[]{1, 1}, new int[]{1, 1}, new int[]{2, 2}, new int[]{2, 2}, 832, 832, "mixed5a"),
  Inc5H_5b(new int[]{0, 0}, new int[]{0, 0}, new int[]{1, 1}, new int[]{1, 1}, 832, 1024, "mixed5b");

  private static transient Map<String, PipelineNetwork> inception5h = null;
  private static volatile VisionPipeline<Inception5H> visionPipeline = null;
  private final String layerId;
  private final int[] inputBorders;
  private final int[] outputBorders;
  private final int[] kenelSize;
  private final int[] strides;
  private final int inputChannels;
  private final int outputChannels;

  Inception5H(int[] inputBorders, int[] outputBorders, int[] kenelSize, int[] strides, int inputChannels, int outputChannels, String layerId) {
    this.inputChannels = inputChannels;
    this.outputChannels = outputChannels;
    this.layerId = layerId;
    this.inputBorders = inputBorders;
    this.outputBorders = outputBorders;
    this.kenelSize = kenelSize;
    this.strides = strides;
  }

  public static Map<String, PipelineNetwork> layerMap() {
    if (null == inception5h) {
      synchronized (Inception5H.class) {
        if (null == inception5h) {
          inception5h = ImageArtUtil.convertPipeline(ImageNetworkPipeline.loadGraphZip(
              "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip",
              "tensorflow_inception_graph.pb"
              ), "conv2d0",
              "localresponsenorm1",
              "mixed3a",
              "mixed3b",
              "mixed4a",
              "mixed4b",
              "mixed4c",
              "mixed4d",
              "mixed4e",
              "mixed5a",
              "mixed5b");
        }
      }
    }
    return inception5h;
  }

  public static VisionPipeline<Inception5H> getVisionPipeline() {
    if (null == visionPipeline) {
      synchronized (Inception5H.class) {
        if (null == visionPipeline) {
          visionPipeline = new VisionPipeline<>(Inception5H.class.getSimpleName(), Inception5H.values());
        }
      }
    }
    return visionPipeline;
  }

  @Override
  public PipelineNetwork getLayer() {
    return (PipelineNetwork) layerMap().get(this.layerId).copyPipeline().setName(name());
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
  public String getPipelineName() {
    return getVisionPipeline().name;
  }

  @Override
  public VisionPipeline<?> getPipeline() {
    return getVisionPipeline();
  }

}
