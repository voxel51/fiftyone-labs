import {
  executeOperator,
} from "@fiftyone/operators";
import * as fos from "@fiftyone/state";
import _ from "lodash";
import { useRecoilValue } from "recoil";
import styled from "styled-components";
import { useState, useRef, useCallback, useEffect } from "react";

export function ClickSegmentation() {
  const modalSample = useRecoilValue(fos.modalSample);
  const [clicks, setClicks] = useState([]);
  const [isCapturing, setIsCapturing] = useState(false);
  const [isSingleClick, setSingleClick] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [fieldName, setFieldName] = useState("user_clicks");
  const [modelName, setModelName] = useState("segment-anything-2-hiera-tiny-image-torch");
  const [labelName, setLabelName] = useState("label");

  useEffect(() => {
    setClicks([]);
  }, [modalSample?.sample?._id]);

  useEffect(() => {
    if (!isCapturing || !modalSample) return;

    const handleCanvasClick = async (event: MouseEvent) => {
      const canvas = event.target as HTMLCanvasElement;
      if (canvas.tagName !== "CANVAS") return;    
      if (isProcessing) return;

      const rect = canvas.getBoundingClientRect();
      const metadata = modalSample.sample.metadata;
    
      if (!metadata?.width || !metadata?.height) {
        console.error("No image metadata available");
        return;
      }

      // Calculate the relative position of the click
      // NOTE: Sample modal displays the image in contain mode filling one dimension while keeping the aspect
      // ratio. It is possible to zoom in and out in the model which will break the correct image coord capture.
      // TODO: Fix this for when image width and height are both smaller than that of modal canvas.
      const imageAspect = metadata.width / metadata.height;
      const canvasAspect = rect.width / rect.height;
    
      let imgWidth, imgHeight, imgOffsetX, imgOffsetY;
    
      if (imageAspect > canvasAspect) {
        // Image fills width only
        imgWidth = rect.width;
        imgHeight = rect.width / imageAspect;
        imgOffsetX = 0;
        imgOffsetY = (rect.height - imgHeight) / 2;
      } else {
        // Image fills height
        imgHeight = rect.height;
        imgWidth = rect.height * imageAspect;
        imgOffsetX = (rect.width - imgWidth) / 2;
        imgOffsetY = 0;
      }
    
      const canvasX = event.clientX - rect.left;
      const canvasY = event.clientY - rect.top;
      const imgX = canvasX - imgOffsetX;
      const imgY = canvasY - imgOffsetY;

      if (imgX < 0 || imgX > imgWidth || imgY < 0 || imgY > imgHeight) {
        console.log("Click outside image bounds");
        return;
      }
    
      const normalizedX = imgX / imgWidth;
      const normalizedY = imgY / imgHeight;    
      const clickData = {
        normalizedX: parseFloat(normalizedX.toFixed(4)),
        normalizedY: parseFloat(normalizedY.toFixed(4)),
      };
    
      console.log("Click captured:", clickData);
      if (isSingleClick) {
        await autoSegmentWithSingleKeypoint(clickData);
      } else {
        setClicks((prev) => [...prev, clickData]);
      }
    };

    document.addEventListener("click", handleCanvasClick, true);
    return () => document.removeEventListener("click", handleCanvasClick, true);
  }, [isCapturing, modalSample, isSingleClick, isProcessing, modelName, fieldName]);

  const saveAsKeypoints = async () => {
    if (!fieldName.trim()) {
      alert("Please enter a field name");
      return;
    }

    const keypointCoords = clicks.map(click => [
      click.normalizedX,
      click.normalizedY,
    ]);
    
    try {
      await executeOperator(
        "@51labs/click_segmentation/save_keypoints",
        {
          keypoints: keypointCoords,
          field_name: fieldName.trim(),
          label_name: labelName.trim()
        }
      );
      setClicks([]);
    } catch (error) {
      console.error("Error saving keypoints:", error);
      alert(`Failed: ${error.message}`);
    }
  };

  const autoSegmentWithSingleKeypoint = async (clickData) => {
    setIsProcessing(true);

    const keypointCoords = [[clickData.normalizedX, clickData.normalizedY]];
    try {
      console.log("Auto segmenting with ", clickData);
      await executeOperator(
        "@51labs/click_segmentation/save_keypoints",
        {
          keypoints: keypointCoords,
          field_name: fieldName.trim(),
          label_name: labelName.trim()
        }
      );
      // Add a delay to ensure keypoints are available
      // TODO: Remove delay. Find a better fix.
      await new Promise(resolve => setTimeout(resolve, 10));
      await executeOperator(
        "@51labs/click_segmentation/segment_with_prompts",
        {
          prompt_field: fieldName.trim(),
          model_name: modelName.trim()
        }
      );
      setClicks([])
    } catch (error) {
      console.error("Error auto segmenting:", error);
      alert(`Failed: ${error.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const segmentWithKeypoints = async () => {
    try {
      await executeOperator(
        "@51labs/click_segmentation/segment_with_prompts",
        {
          prompt_field: fieldName.trim(),
          model_name: modelName.trim()
        }
      );
    } catch (error) {
      console.error("Error saving keypoints:", error);
      alert(`Failed: ${error.message}`);
    }
  };

  const clearClicks = () => {
    setClicks([]);
  };
  
  return (
    <div style={{ padding: "20px", display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ marginBottom: "10px" }}>
        <h3 style={{ margin: 0 }}>Image segmentation via point prompts</h3>
      </div>
      
      {/* Capture toggle on/off*/}
      <div style={{ 
        marginBottom: "20px",
        display: "flex",
        flexDirection: "column",
        padding: "15px",
        backgroundColor: isCapturing ? "#e3f2fd" : "#f5f5f5",
        borderRadius: "8px",
        border: `2px solid ${isCapturing ? "#2196F3" : "#ddd"}`,
        transition: "all 0.3s ease"
      }}>
        <label style={{ 
          display: "flex", 
          alignItems: "center", 
          gap: "10px", 
          cursor: "pointer" 
        }}>
          <input 
            type="checkbox"
            checked={isCapturing}
            onChange={(e) => setIsCapturing(e.target.checked)}
            style={{ 
              width: "20px", 
              height: "20px", 
              cursor: "pointer" 
            }}
          />
          <span style={{ 
            fontWeight: "bold", 
            fontSize: "15px",
            color: isCapturing ? "#1976d2" : "#666"
          }}>
            {isCapturing ? "Capturing Clicks" : "Activate Click Capture"}
          </span>
        </label>
        {isCapturing && (
          <p style={{ 
            fontSize: "12px", 
            color: "#1976d2", 
            margin: "8px 0 0 30px",
            fontWeight: "500"
          }}>
            Click anywhere on the modal image to capture keypoints.
          </p>
        )}
      </div>

      {/* Field name input */}
      <div style={{ 
        marginBottom: "8px", 
        padding: "4px",
      }}>
        <label style={{ 
          display: "flex", 
          flexDirection: "column", 
          gap: "8px",
          fontSize: "14px"
        }}>
          <span style={{ fontSize: "14px", color: "#666" }}>
            Sample field name for saving clicks (as keypoints)
          </span>
          <input 
            type="text"
            value={fieldName}
            onChange={(e) => setFieldName(e.target.value)}
            placeholder="e.g., user_clicks, keypoints"
            style={{
              padding: "8px 12px",
              border: "1px solid #ccc",
              borderRadius: "4px",
              fontSize: "14px",
              fontFamily: "monospace"
            }}
          />
        </label>
      </div>

      {/* Keypoint label name input */}
      <div style={{ 
        marginBottom: "8px", 
        padding: "4px",
      }}>
        <label style={{ 
          display: "flex", 
          flexDirection: "column", 
          gap: "8px",
          fontSize: "14px"
        }}>
          <span style={{ fontSize: "14px", color: "#666" }}>
            Label name for the current set of clicks
          </span>
          <input 
            type="text"
            value={labelName}
            onChange={(e) => setLabelName(e.target.value)}
            placeholder="e.g., animal, person"
            style={{
              padding: "8px 12px",
              border: "1px solid #ccc",
              borderRadius: "4px",
              fontSize: "14px",
              fontFamily: "monospace"
            }}
          />
        </label>
      </div>

      {/* Single click execution */}
      <div style={{ 
        marginBottom: "20px",
        display: "flex",
        flexDirection: "column",
        padding: "15px",
        backgroundColor: isSingleClick ? "#e3f2fd" : "#f5f5f5",
        borderRadius: "8px",
        border: `2px solid ${isSingleClick ? "#2196F3" : "#ddd"}`,
        transition: "all 0.3s ease"
      }}>
        <label style={{ 
          display: "flex", 
          alignItems: "center", 
          gap: "10px", 
          cursor: "pointer" 
        }}>
          <input 
            type="checkbox"
            checked={isSingleClick}
            onChange={(e) => setSingleClick(e.target.checked)}
            style={{ 
              width: "20px", 
              height: "20px", 
              cursor: "pointer" 
            }}
          />
          <span style={{ 
            fontWeight: "bold", 
            fontSize: "15px",
            color: isSingleClick ? "#1976d2" : "#666"
          }}>
            {isSingleClick ? "Single Click Segmentation" : "Enable Single Click Segmentation"}
          </span>
        </label>
        {isSingleClick && (
          <p style={{ 
            fontSize: "12px", 
            color: "#1976d2", 
            margin: "8px 0 0 30px",
            fontWeight: "500"
          }}>
            Single click segmentation is enabled. Clicking on the modal will auto-generate segmentation.
          </p>
        )}
      </div>

      {/* Keypoint buttons */}
      <div style={{ marginBottom: "10px", display: "flex", gap: "10px" }}>
      <button 
          onClick={saveAsKeypoints}
          disabled={(isSingleClick) || (clicks.length === 0)}
          style={{
            padding: "8px 16px",
            backgroundColor: (!isSingleClick) && (clicks.length > 0) ? "#2196F3" : "#ccc",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: (!isSingleClick) && (clicks.length > 0) ? "pointer" : "not-allowed",
            fontWeight: "bold"
          }}
        >
          Save as Keypoints ({clicks.length})
        </button>
        <button 
          onClick={clearClicks}
          disabled={(isSingleClick) || (clicks.length === 0)}
          style={{
            padding: "8px 16px",
            backgroundColor: (!isSingleClick) || (clicks.length > 0) ? "#2196F3" : "#ccc",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: (!isSingleClick) && (clicks.length > 0) ? "pointer" : "not-allowed",
          }}
        >
          Clear Clicks ({clicks.length})
        </button>
      </div>

      {/* Model name input */}
      <div style={{ 
        marginBottom: "8px", 
        padding: "4px",
      }}>
        <label style={{ 
          display: "flex", 
          flexDirection: "column", 
          gap: "8px",
          fontSize: "14px"
        }}>
          <span style={{ fontSize: "14px", color: "#666" }}>
            Promptable segmentation model from FiftyOne model zoo
          </span>
          <input 
            type="text"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            placeholder="e.g., segment-anything-2-hiera-small-image-torch"
            style={{
              padding: "8px 12px",
              border: "1px solid #ccc",
              borderRadius: "4px",
              fontSize: "14px",
              fontFamily: "monospace"
            }}
          />
        </label>
      </div>

      {/* Segment with keypoints button */}
      <div style={{ marginBottom: "10px", display: "flex", gap: "10px" }}>
        <button 
          onClick={segmentWithKeypoints}
          disabled={isSingleClick}
          style={{
            padding: "8px 16px",
            backgroundColor: (!isSingleClick) ? "#2196F3" : "#ccc",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: (!isSingleClick) ? "pointer" : "not-allowed",
            fontWeight: "bold"
          }}
        >
          Segment with keypoints
        </button>
      </div>
    </div>
  );
}