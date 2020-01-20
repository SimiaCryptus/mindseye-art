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

import com.simiacryptus.ref.wrappers.RefArrays;

import javax.annotation.Nonnull;
import java.awt.*;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;

public class TextUtil {
  @Nonnull
  public static BufferedImage draw(@Nonnull final String text, final int resolution, final int padding, final String fontName,
                                   final int style) {
    return draw(text, resolution, padding, fitWidth(text, resolution, padding, fontName, style));
  }

  @Nonnull
  public static BufferedImage draw(@Nonnull String text, int resolution, int padding, Font font) {
    Rectangle2D bounds = measure(font, text);
    double aspect_ratio = (2.0 * padding + bounds.getHeight()) / (2.0 * padding + bounds.getWidth());
    return draw(text, resolution, (int) (aspect_ratio * resolution), padding, font, bounds);
  }

  @Nonnull
  public static BufferedImage drawHeight(@Nonnull String text, int resolution, int padding, Font font) {
    Rectangle2D bounds = measure(font, text);
    double aspect_ratio = (2.0 * padding + bounds.getHeight()) / (2.0 * padding + bounds.getWidth());
    return draw(text, (int) (resolution / aspect_ratio), resolution, padding, font, bounds);
  }

  public @Nonnull
  static BufferedImage draw(@Nonnull String text, int width, int height, int padding, Font font,
                            @Nonnull Rectangle2D bounds) {
    BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
    Graphics2D graphics = (Graphics2D) image.getGraphics();
    graphics.setColor(Color.WHITE);
    graphics.setFont(font);
    double initY = ((Rectangle2D.Double) bounds).y + padding;
    int y = (int) initY;
    for (final String line : text.split("\n")) {
      Rectangle2D stringBounds = graphics.getFontMetrics().getStringBounds(line, graphics);
      y += stringBounds.getHeight();
    }
    y = (int) (initY + (((height - padding) - (y - initY)) / 2));
    for (final String line : text.split("\n")) {
      Rectangle2D stringBounds = graphics.getFontMetrics().getStringBounds(line, graphics);
      double centeringOffset = (bounds.getWidth() - stringBounds.getWidth()) / 2;
      graphics.drawString(line, (int) (padding + centeringOffset), y);
      y += stringBounds.getHeight();
    }
    return image;
  }

  @Nonnull
  public static Rectangle2D measure(final Font font, @Nonnull final String text) {
    Graphics2D graphics = (Graphics2D) new BufferedImage(100, 100, BufferedImage.TYPE_INT_RGB).getGraphics();
    graphics.setFont(font);
    String[] lines = text.split("\n");
    double width = RefArrays.stream(lines)
        .mapToInt(t -> (int) graphics.getFontMetrics().getStringBounds(t, graphics).getWidth()).max().getAsInt();
    int height = RefArrays.stream(lines)
        .mapToInt(t -> (int) graphics.getFontMetrics().getLineMetrics(t, graphics).getAscent()).sum();
    double line1height = graphics.getFontMetrics().getLineMetrics(lines[0], graphics).getAscent();
    return new Rectangle2D.Double(0, line1height, width, height);
  }

  @Nonnull
  public static Font fitWidth(@Nonnull final String text, final int resolution, final int padding, final String fontName,
                              final int style) {
    Graphics2D graphics = (Graphics2D) new BufferedImage(100, 100, BufferedImage.TYPE_INT_RGB).getGraphics();
    double width = 0;
    int size = 12;
    while (width < (resolution - 2 * padding) && size < 1000) {
      size += 2;
      graphics.setFont(new Font(fontName, style, size));
      width = RefArrays.stream(text.split("\n"))
          .mapToInt(t -> (int) graphics.getFontMetrics().getStringBounds(t, graphics).getWidth()).max().getAsInt();
    }
    size -= 2;
    final Font font = new Font(fontName, style, size);
    return font;
  }

  @Nonnull
  public static Font fitHeight(@Nonnull final String text, final int resolution, final int padding, final String fontName,
                               final int style) {
    double height = 0;
    int size = 12;
    while (height < (resolution - 2 * padding) && size < 10000) {
      size += 2;
      height = measure(new Font(fontName, style, size), text).getHeight();
    }
    size -= 2;
    final Font font = new Font(fontName, style, size);
    return font;
  }

  @Nonnull
  public static Font fit(@Nonnull final String text, final int max_width, final int max_height, final int padding,
                         final String fontName, final int style) {
    double height = 0;
    double width = 0;
    int size = 12;
    while (height < (max_height - 2 * padding) && width < (max_width - 2 * padding) && size < 10000) {
      size += 2;
      Rectangle2D measure = measure(new Font(fontName, style, size), text);
      height = measure.getHeight();
      width = measure.getWidth();
    }
    size -= 2;
    final Font font = new Font(fontName, style, size);
    return font;
  }
}
