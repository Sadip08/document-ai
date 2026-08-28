import cv2
import pytesseract
import pandas as pd

def parse_table(image, gap_threshold=50):
    """Extracts tabular data from a cropped table image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config='--psm 6')
    
    tokens = []
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])
        if text and conf > 0:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            tokens.append({'text': text, 'x1': x, 'y1': y, 'x2': x+w, 'y2': y+h, 'center_y': y + h/2})
            
    if not tokens:
        return None
        
    # Group into rows
    tokens.sort(key=lambda t: t['y1'])
    rows = []
    current_row = [tokens[0]]
    for token in tokens[1:]:
        avg_y = sum([t['center_y'] for t in current_row]) / len(current_row)
        if abs(token['center_y'] - avg_y) < 15:
            current_row.append(token)
        else:
            rows.append(current_row)
            current_row = [token]
    rows.append(current_row)
    
    for row in rows:
        row.sort(key=lambda t: t['x1'])
        
    # Group into columns based on gap
    table_data = []
    for row in rows:
        row_cells = []
        current_cell_text = [row[0]['text']]
        for i in range(1, len(row)):
            gap = row[i]['x1'] - row[i-1]['x2']
            if gap > gap_threshold:
                row_cells.append(" ".join(current_cell_text))
                current_cell_text = [row[i]['text']]
            else:
                current_cell_text.append(row[i]['text'])
        row_cells.append(" ".join(current_cell_text))
        table_data.append(row_cells)
        
    max_cols = max(len(row) for row in table_data)
    for row in table_data:
        while len(row) < max_cols:
            row.append("")
            
    df = pd.DataFrame(table_data[1:], columns=table_data[0])
    return df