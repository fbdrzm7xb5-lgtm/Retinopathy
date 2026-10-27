package com.retinagrade.backend.service;

import com.retinagrade.backend.exception.InvalidImageException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Locale;

@Service
public class ImageValidationService {

    private static final Logger log = LoggerFactory.getLogger(ImageValidationService.class);

    private static final long MAX_FILE_SIZE_BYTES = 10L * 1024L * 1024L; // 10 MB
    private static final int MIN_DIMENSION = 64; // Minimum 64x64 resolution for retinal screening

    // Standard Magic Bytes
    // JPEG: FF D8 FF
    private static final byte[] JPEG_MAGIC = new byte[]{(byte) 0xFF, (byte) 0xD8, (byte) 0xFF};
    // PNG: 89 50 4E 47 0D 0A 1A 0A
    private static final byte[] PNG_MAGIC = new byte[]{(byte) 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};

    /**
     * Validates an uploaded retinal image for presence, size, MIME type, extension, magic bytes, and decodability.
     *
     * @param file the multipart file
     * @return the decoded BufferedImage for downstream preprocessing
     * @throws InvalidImageException if any validation rule is violated
     */
    public BufferedImage validateAndRead(MultipartFile file) {
        if (file == null || file.isEmpty()) {
            throw new InvalidImageException("No file was uploaded. Please provide a retinal fundus image.");
        }

        // 1. File Size Validation
        if (file.getSize() > MAX_FILE_SIZE_BYTES) {
            double sizeInMb = file.getSize() / (1024.0 * 1024.0);
            throw new InvalidImageException(String.format(
                    Locale.US,
                    "File size (%.2f MB) exceeds the 10 MB limit. Please upload an image under 10 MB.",
                    sizeInMb
            ));
        }

        // 2. Filename & Extension Validation
        String originalFilename = file.getOriginalFilename();
        if (originalFilename == null || originalFilename.isBlank()) {
            throw new InvalidImageException("Uploaded file must have a valid filename with an extension.");
        }

        String lowerName = originalFilename.toLowerCase(Locale.ROOT);
        boolean hasValidExtension = lowerName.endsWith(".jpg") ||
                                    lowerName.endsWith(".jpeg") ||
                                    lowerName.endsWith(".png");
        if (!hasValidExtension) {
            throw new InvalidImageException("Invalid file extension: '" + originalFilename + "'. Only JPG, JPEG, and PNG formats are accepted.");
        }

        // 3. MIME Content-Type Validation
        String contentType = file.getContentType();
        if (contentType != null && !contentType.isBlank()) {
            String lowerType = contentType.toLowerCase(Locale.ROOT);
            boolean isValidMime = lowerType.equals("image/jpeg") ||
                                  lowerType.equals("image/pjpeg") ||
                                  lowerType.equals("image/png");
            if (!isValidMime) {
                throw new InvalidImageException("Invalid Content-Type header: '" + contentType + "'. Expected 'image/jpeg' or 'image/png'.");
            }
        }

        // 4. Magic Bytes Inspection (Binary Header Verification)
        verifyMagicBytes(file);

        // 5. Image Decoding & Dimension Validation
        try (InputStream is = file.getInputStream()) {
            BufferedImage image = ImageIO.read(is);
            if (image == null) {
                throw new InvalidImageException("The uploaded file could not be decoded as a valid image. The data may be corrupt.");
            }

            if (image.getWidth() < MIN_DIMENSION || image.getHeight() < MIN_DIMENSION) {
                throw new InvalidImageException(String.format(
                        "Image resolution (%dx%d) is too low for reliable retinal analysis. Minimum required is %dx%d.",
                        image.getWidth(), image.getHeight(), MIN_DIMENSION, MIN_DIMENSION
                ));
            }

            log.debug("Image validated successfully: {} ({}x{}, {} bytes)",
                    originalFilename, image.getWidth(), image.getHeight(), file.getSize());

            return image;
        } catch (IOException e) {
            throw new InvalidImageException("Failed to read image stream: " + e.getMessage(), e);
        }
    }

    private void verifyMagicBytes(MultipartFile file) {
        try (InputStream is = new BufferedInputStream(file.getInputStream())) {
            byte[] header = new byte[8];
            int read = is.read(header, 0, header.length);
            if (read < 3) {
                throw new InvalidImageException("Uploaded file is too small to contain a valid image header.");
            }

            boolean isJpeg = header[0] == JPEG_MAGIC[0] &&
                             header[1] == JPEG_MAGIC[1] &&
                             header[2] == JPEG_MAGIC[2];

            boolean isPng = read >= 8 &&
                            header[0] == PNG_MAGIC[0] &&
                            header[1] == PNG_MAGIC[1] &&
                            header[2] == PNG_MAGIC[2] &&
                            header[3] == PNG_MAGIC[3] &&
                            header[4] == PNG_MAGIC[4] &&
                            header[5] == PNG_MAGIC[5] &&
                            header[6] == PNG_MAGIC[6] &&
                            header[7] == PNG_MAGIC[7];

            if (!isJpeg && !isPng) {
                throw new InvalidImageException("File binary signature does not match standard JPEG or PNG magic bytes. Renamed non-image files are rejected.");
            }
        } catch (IOException e) {
            throw new InvalidImageException("Unable to inspect file binary header: " + e.getMessage(), e);
        }
    }
}

