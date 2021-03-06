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
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.wrappers.RefMap;
import com.simiacryptus.tensorflow.ImageNetworkPipeline;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

/**
 * The enum Inception 5 h.
 */
public enum Inception5H implements VisionPipelineLayer {
  /**
   * Inc 5 h 1 a inception 5 h.
   */
  Inc5H_1a("conv2d0"),
  /**
   * Inc 5 h 2 a inception 5 h.
   */
  Inc5H_2a("localresponsenorm1"),
  /**
   * Inc 5 h 3 a inception 5 h.
   */
  Inc5H_3a("mixed3a"),
  /**
   * Inc 5 h 3 b inception 5 h.
   */
  Inc5H_3b("mixed3b"),
  /**
   * Inc 5 h 4 a inception 5 h.
   */
  Inc5H_4a("mixed4a"),
  /**
   * Inc 5 h 4 b inception 5 h.
   */
  Inc5H_4b("mixed4b"),
  /**
   * Inc 5 h 4 c inception 5 h.
   */
  Inc5H_4c("mixed4c"),
  /**
   * Inc 5 h 4 d inception 5 h.
   */
  Inc5H_4d("mixed4d"),
  /**
   * Inc 5 h 4 e inception 5 h.
   */
  Inc5H_4e("mixed4e"),
  /**
   * Inc 5 h 5 a inception 5 h.
   */
  Inc5H_5a("mixed5a"),
  /**
   * Inc 5 h 5 b inception 5 h.
   */
  Inc5H_5b("mixed5b");

  @Nullable
  private static transient RefMap<String, PipelineNetwork> inception5h = null;
  @Nullable
  private static volatile VisionPipeline visionPipeline = null;
  private final String layerId;

  Inception5H(String layerId) {
    this.layerId = layerId;
  }

  @Nonnull
  @Override
  public PipelineNetwork getLayer() {
    RefMap<String, PipelineNetwork> layerMap = layerMap();
    PipelineNetwork network = layerMap.get(this.layerId);
    layerMap.freeRef();
    Layer layer = network.copyPipeline();
    network.freeRef();
    layer.setName(name());
    return (PipelineNetwork) layer;
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
      synchronized (Inception5H.class) {
        if (null == visionPipeline) {
          visionPipeline = new VisionPipeline(Inception5H.class.getSimpleName(), Inception5H.values());
        }
      }
    }
    return visionPipeline.addRef();
  }

  /**
   * Layer map ref map.
   *
   * @return the ref map
   */
  @Nonnull
  public static RefMap<String, PipelineNetwork> layerMap() {
    if (null == inception5h) {
      synchronized (Inception5H.class) {
        if (null == inception5h) {
          inception5h = ImageArtUtil.convertPipeline(
              ImageNetworkPipeline.loadGraphZip(
                  "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip",
                  "tensorflow_inception_graph.pb"),
              "conv2d0", "localresponsenorm1", "mixed3a", "mixed3b", "mixed4a", "mixed4b", "mixed4c", "mixed4d",
              "mixed4e", "mixed5a", "mixed5b");
        }
      }
    }
    return inception5h.addRef();
  }

}
