import cv2
import numpy as np
from depthai_sdk import OakCamera
from depthai_sdk.classes import TwoStagePacket
from depthai_sdk.visualize.configs import TextPosition


class PedestrianReId:
    def __init__(self):
        self.results = []

    def _cosine_dist(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def new_result(self, vector_result):
        vector_result = np.array(vector_result)
        for i, vector in enumerate(self.results):
            dist = self._cosine_dist(vector, vector_result)
            if dist > 0.7:
                self.results[i] = vector_result
                return i
        else:
            self.results.append(vector_result)
            return len(self.results) - 1


with OakCamera() as oak:
    color = oak.create_camera('color', fps=10)
    person_det = oak.create_nn('person-detection-retail-0013', input=color)
    person_det.node.setNumInferenceThreads(2)
    person_det.config_nn(resize_mode='crop')

    nn_reid = oak.create_nn('person-reidentification-retail-0288', input=person_det)
    nn_reid.node.setNumInferenceThreads(2)

    reid = PedestrianReId()
    results = []

    bounding_box_color_green = (0, 255, 0)  # Green color
    bounding_box_color_yellow = (0, 255, 255)  # Yellow color
    bounding_box_color_red = (0, 0, 255)  # Red color

    def cb(packet: TwoStagePacket):
        visualizer = packet.visualizer

        # create a mask with a red bar in the center
        mask = np.zeros_like(packet.frame)
        h, w = packet.frame.shape[:2]
        bar_width = w // 4  # adjust this value to change the width of the bar
        x = w // 2 - bar_width // 2
        cv2.rectangle(mask, (x, 0), (x + bar_width, h), (0, 0, 255), thickness=-1)

        # blend the mask with the original image without changing its color
        alpha = 0.3  # opacity of the mask
        frame = cv2.addWeighted(packet.frame, 1 - alpha, mask, alpha, 0)
        danger_zone_count = 0 
        # check if any person is in the danger zone, close to the danger zone, or in the safe zone
        for det, rec in zip(packet.detections, packet.nnData):
            reid_result = rec.getFirstLayerFp16()
            id = reid.new_result(reid_result)

            # Count the number of people boxes
            num_people = len(packet.detections)

            # Add the count to the top right of the frame
            

            # measure the distance between the person and the danger zone
            danger_center = (x + bar_width // 2, h // 2)
            #person_center = ((det.top_left[0] + det.bottom_right[0]) // 2,
             #    det.bottom_right[1])
            person_center = ((det.top_left[0] + det.bottom_right[0]) // 2, (det.top_left[1] + det.bottom_right[1]) // 2)

            distance = np.linalg.norm(np.array(danger_center) - np.array(person_center))

            if distance < 200:
                visualizer.add_text("Danger", bbox=(*det.top_left, *det.bottom_right),
                                    color=(0, 0, 255), position=TextPosition.MID)
                bounding_box_color = bounding_box_color_red  # Red color
                danger_zone_count += 1
            elif distance < 350:
                visualizer.add_text("Close to Danger Zone", bbox=(*det.top_left, *det.bottom_right),
                                    color=(0, 255, 255), position=TextPosition.MID)
                bounding_box_color = bounding_box_color_yellow  # Yellow color
            else:
                visualizer.add_text("Safe", bbox=(*det.top_left, *det.bottom_right),
                                    color=(0, 255, 0), position=TextPosition.MID)
                bounding_box_color = bounding_box_color_green  # Green color
            text = f"Total People: {danger_zone_count}"
            cv2.putText(frame, text, (w - 260, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            box_width = det.bottom_right[0] - det.top_left[0]
            box_height = det.bottom_right[1] - det.top_left[1]
            cv2.rectangle(frame, (det.top_left[0], det.top_left[1]), (det.top_left[0] + box_width, det.top_left[1] + box_height),
                        bounding_box_color, 2)
            center_x = int((det.bottom_right[0] + det.top_left[0]) / 2)
            center_y = int((det.bottom_right[1] + det.top_left[1]) / 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)


        # draw the frame with visualizations
        frame = visualizer.draw(frame)
        cv2.imshow('Person reidentification', frame)

        # Exit condition: Press 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            oak.stop()


    oak.visualize(nn_reid, callback=cb, fps=True)
    oak.start(blocking=True)