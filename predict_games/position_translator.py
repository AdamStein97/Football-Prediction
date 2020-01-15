import os

class Translator():
    def __init__(self):
        self.heatmaps = []
        self.position_dict = {
            "cb": "Center Back",
            "rb": "Right Back",
            "lb": "Left Back",
            "cm": "Center Midfield",
            "rm": "Right Midfield",
            "lm": "Left Midfield",
            "dm": "Defensive Midfield",
            "am": "Attacking Midfield",
            "rw": "Right Wing",
            "lw": "Left Wing",
            "fw": "Center Forward"
        }
        self.load_heatmaps()

    def load_heatmaps(self):
        for data in os.listdir("../data/scaled_heatmaps"):
            self.heatmaps.append(data[:-4])

    def translate_position(self,pos_ab,formation):
        position = self.position_dict[pos_ab[:2]]
        modifier = pos_ab[-1]
        if position in ["Center Back","Center Midfield","Defensive Midfield","Attacking Midfield","Center Forward"]:
            left = "Left " + position + "_" + formation
            centre = position + "_" + formation
            right = "Right " + position + "_" + formation
            if modifier == "1":
                if left in self.heatmaps:
                    return left
                elif centre in self.heatmaps:
                    return centre
                # else:
                #     print("Translation failed for: " + pos_ab + "_" + formation)
                #     return ""
            if modifier == "2":
                if centre in self.heatmaps:
                    return centre
                elif right in self.heatmaps:
                    return right
                # else:
                #     print("Translation failed for: " + pos_ab + "_" + formation)
                #     return ""
            if modifier == "3":
                if right in self.heatmaps:
                    return right
                elif centre in self.heatmaps:
                    return centre
                # else:
                #     print("Translation failed for: " + pos_ab + "_" + formation)
                #     return ""
        if position == "Right Back" or position == "Left Back":
            fullback = position + "_" + formation
            wingback = position.split(" ")[0] + " Wing Back" + "_" + formation
            if fullback in self.heatmaps:
                return fullback
            elif wingback in self.heatmaps:
                return wingback
            # else:
            #     print("Translation failed for: " + pos_ab + "_" + formation)
            #     return ""
        if position == "Right Wing Back" or position == "Left Wing Back":
            fullback = position.split(" ")[0] + " Back" + "_" + formation
            wingback = position + "_" + formation
            if wingback in self.heatmaps:
                return wingback
            elif fullback in self.heatmaps:
                return fullback
            # else:
            #     print("Translation failed for: " + pos_ab + "_" + formation)
            #     return ""
        if position == "Right Midfield" or position == "Left Midfield":
            midfield = position + "_" + formation
            wing = position.split(" ")[0] + " Wing" + "_" + formation
            if midfield in self.heatmaps:
                return midfield
            elif wing in self.heatmaps:
                return wing
            # else:
            #     print("Translation failed for: " + pos_ab + "_" + formation)
            #     return ""
        if position == "Right Wing" or position == "Left Wing":
            midfield = position.split(" ")[0] + " Midfield" + "_" + formation
            wing = position + "_" + formation
            if wing in self.heatmaps:
                return wing
            elif midfield in self.heatmaps:
                return midfield
            # else:
            #     print("Translation failed for: " + pos_ab + "_" + formation)
            #     return ""
        # print("Translation failed for: " + pos_ab + "_" + formation + " returning default...")
        return [heatmap for heatmap in self.heatmaps if position in heatmap][0]

