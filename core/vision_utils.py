import numpy as np

class ContextBuilder:
    """
    Builds spatial relationship context from YOLO detections.
    Logic transferred directly from Colab notebook.
    """

    def __init__(self,
                 on_threshold=0.3,
                 near_threshold=150,
                 horizontal_alignment_threshold=50):
        self.on_threshold = on_threshold
        self.near_threshold = near_threshold
        self.horizontal_alignment_threshold = horizontal_alignment_threshold

        self.surface_objects = {'table', 'desk', 'bed', 'couch', 'chair',
                               'dining table', 'counter', 'shelf'}

        self.holdable_objects = {'cell phone', 'bottle', 'cup', 'book',
                                'remote', 'fork', 'knife', 'spoon', 'umbrella'}

    def get_bbox_center(self, bbox):
        return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

    def calculate_distance(self, center1, center2):
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def is_on(self, obj1, obj2):
        if obj2['class_name'] not in self.surface_objects:
            return False

        obj1_bottom = obj1['bbox'][3]
        obj2_top = obj2['bbox'][1]

        # Simple heuristic for "on top"
        vertical_aligned = abs(obj1_bottom - obj2_top) < 50

        obj1_center_x = (obj1['bbox'][0] + obj1['bbox'][2]) / 2
        obj2_left = obj2['bbox'][0]
        obj2_right = obj2['bbox'][2]
        horizontal_overlap = obj2_left <= obj1_center_x <= obj2_right

        return vertical_aligned and horizontal_overlap

    def get_horizontal_relationship(self, obj1, obj2):
        center1 = self.get_bbox_center(obj1['bbox'])
        center2 = self.get_bbox_center(obj2['bbox'])

        vertical_diff = abs(center1[1] - center2[1])
        if vertical_diff > self.horizontal_alignment_threshold:
            return None

        if center1[0] < center2[0]:
            return "left_of"
        else:
            return "right_of"

    def is_holding(self, person, obj):
        if person['class_name'] != 'person':
            return False

        if obj['class_name'] not in self.holdable_objects:
            return False

        person_box = person['bbox']
        obj_center = self.get_bbox_center(obj['bbox'])

        in_horizontal = person_box[0] <= obj_center[0] <= person_box[2]

        person_height = person_box[3] - person_box[1]
        upper_region = person_box[1] + (person_height * 0.7)
        in_upper = obj_center[1] <= upper_region

        return in_horizontal and in_upper

    def build_relationships(self, detections):
        relationships = []

        # Sort by area (largest first) to handle "small object on big object"
        sorted_detections = sorted(detections,
                                   key=lambda x: (x['bbox'][2] - x['bbox'][0]) *
                                               (x['bbox'][3] - x['bbox'][1]),
                                   reverse=True)

        for i, obj1 in enumerate(sorted_detections):
            for j, obj2 in enumerate(sorted_detections):
                if i == j:
                    continue

                obj1_name = obj1['class_name']
                obj2_name = obj2['class_name']

                if self.is_on(obj1, obj2):
                    relationships.append((obj1_name, "on", obj2_name))
                    continue

                if self.is_holding(obj1, obj2):
                    relationships.append((obj1_name, "holding", obj2_name))
                    continue

                horiz_rel = self.get_horizontal_relationship(obj1, obj2)
                if horiz_rel:
                    relationships.append((obj1_name, horiz_rel, obj2_name))

                center1 = self.get_bbox_center(obj1['bbox'])
                center2 = self.get_bbox_center(obj2['bbox'])
                distance = self.calculate_distance(center1, center2)

                if distance < self.near_threshold:
                    relationships.append((obj1_name, "near", obj2_name))

        # Remove duplicates
        relationships = list(set(relationships))
        return relationships

    def process_frame(self, frame_data, question=None):
        detections = frame_data['detections']
        relationships = self.build_relationships(detections)

        # Format for VLM
        objects_list = [f"{d['class_name']} (confidence: {d['confidence']:.2f})"
                       for d in detections]

        relations_text = []
        for obj1, rel, obj2 in relationships:
            rel_formatted = rel.replace("_", " ")
            relations_text.append(f"- {obj1} is {rel_formatted} {obj2}")

        vlm_prompt = f"""**Scene Analysis:**

**Detected Objects:** {', '.join(objects_list)}

**Spatial Relationships:**
{chr(10).join(relations_text) if relations_text else "- No clear spatial relationships detected"}
"""

        if question:
            vlm_prompt += f"""
**Question:** {question}

Based on the scene analysis above and the visual information, please answer the question accurately."""

        return {
            'frame_id': frame_data.get('frame_id', 0),
            'num_objects': len(detections),
            'objects': [d['class_name'] for d in detections],
            'relationships': relationships,
            'vlm_prompt': vlm_prompt
        }