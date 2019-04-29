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

package com.simiacryptus.mindseye.art.util;

import javax.annotation.Nonnull;
import java.awt.*;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.util.Arrays;

public class TextUtil {
  public static BufferedImage draw(
      final String text,
      final int resolution,
      final int padding,
      final String fontName,
      final int style
  ) {
    Font font = fitSize(text, resolution, padding, fontName, style);
    Rectangle2D bounds = measure(font, text);
    double aspect_ratio = (2.0 * padding + bounds.getHeight()) / (2.0 * padding + bounds.getWidth());
    BufferedImage image = new BufferedImage(resolution, (int) (aspect_ratio * resolution), BufferedImage.TYPE_INT_RGB);
    Graphics2D graphics = (Graphics2D) image.getGraphics();
    graphics.setColor(Color.WHITE);
    graphics.setFont(font);
    int y = (int) ((Rectangle2D.Double) bounds).y + padding;
    for (final String line : text.split("\n")) {
      Rectangle2D stringBounds = graphics.getFontMetrics().getStringBounds(line, graphics);
      double centeringOffset = (bounds.getWidth() - stringBounds.getWidth()) / 2;
      graphics.drawString(line, (int) (padding + centeringOffset), y);
      y += stringBounds.getHeight();
    }
    return image;
  }

  @Nonnull
  public static Rectangle2D measure(final Font font, final String text) {
    final Rectangle2D bounds;
    Graphics2D graphics = (Graphics2D) new BufferedImage(100, 100, BufferedImage.TYPE_INT_RGB).getGraphics();
    graphics.setFont(font);
    String[] lines = text.split("\n");
    double width = Arrays.stream(lines).mapToInt(t -> (int) graphics.getFontMetrics().getStringBounds(t, graphics).getWidth()).max().getAsInt();
    int height = Arrays.stream(lines).mapToInt(t -> (int) graphics.getFontMetrics().getLineMetrics(t, graphics).getAscent()).sum();
    double line1height = graphics.getFontMetrics().getLineMetrics(lines[0], graphics).getAscent();
    bounds = new Rectangle2D.Double(0, line1height, width, height);
    return bounds;
  }

  @Nonnull
  public static Font fitSize(
      final String text,
      final int resolution,
      final int padding,
      final String fontName, final int style
  ) {
    final Font font;
    Graphics2D graphics = (Graphics2D) new BufferedImage(100, 100, BufferedImage.TYPE_INT_RGB).getGraphics();
    double width = 0;
    int size = 12;
    while (width < (resolution - 2 * padding) && size < 1000) {
      size += 2;
      graphics.setFont(new Font(fontName, style, size));
      width = Arrays.stream(text.split("\n")).mapToInt(t -> (int) graphics.getFontMetrics().getStringBounds(
          t,
          graphics
      ).getWidth()).max().getAsInt();
    }
    size -= 2;
    font = new Font(fontName, style, size);
    return font;
  }
}
