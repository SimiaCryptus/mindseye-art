/*
 * Copyright (c) 2020 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.art;

import com.simiacryptus.lang.Settings;
import com.simiacryptus.ref.lang.RefIgnore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nullable;

/**
 * The type Art settings.
 */
@RefIgnore
public class ArtSettings implements Settings {

  private static final Logger logger = LoggerFactory.getLogger(ArtSettings.class);
  @Nullable
  private static transient ArtSettings INSTANCE = null;
  /**
   * The Default tile size.
   */
  public final int defaultTileSize;

  /**
   * Instantiates a new Art settings.
   */
  protected ArtSettings() {
    System.setProperty("java.util.concurrent.ForkJoinPool.common.parallelism", Integer.toString(Settings.get("THREADS", 64)));
    this.defaultTileSize = Settings.get("DEFAULT_TILE_SIZE", 1024);
  }

  /**
   * Instance art settings.
   *
   * @return the art settings
   */
  @Nullable
  public static ArtSettings INSTANCE() {
    if (null == INSTANCE) {
      synchronized (ArtSettings.class) {
        if (null == INSTANCE) {
          INSTANCE = new ArtSettings();
          logger.info(String.format("Initialized %s = %s", INSTANCE.getClass().getSimpleName(), Settings.toJson(INSTANCE)));
        }
      }
    }
    return INSTANCE;
  }
}
