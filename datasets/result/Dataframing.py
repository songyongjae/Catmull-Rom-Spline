import os
import pandas as pd
import re

name = 'set4'


path = f'{name}/labels'
file_list = os.listdir(path)
print(f"총 파일 수 : {len(file_list)}")

df = pd.DataFrame()


for i in file_list:
    file_path = os.path.join(path, i)
    try:
        data = pd.read_csv(file_path, delimiter=' ', encoding='utf-8',header=None)
        print(f"{i} 읽기 완료, 행 수 : {len(data)}")
        match = re.search(r'mp4-(\d+)',i)
        if match:
            number = match.group(1)
            data['File_num'] = int(number)
        
        
        df = pd.concat([df, data], axis=0, ignore_index=False)
    except Exception as e:
        print(f"Error reading {i}: {e}")


df = df.reset_index(drop=True)
df.columns = ['Object','X', 'Y', 'H', 'W','File_num']

cols = df.columns.tolist()
cols = cols[:1] + cols[-1:] + cols[1:-1]
df = df[cols]

df = df.drop(columns=['Object'])

#sorting
df = df.sort_values(by='File_num').reset_index(drop=True)

print(f"최종 데이터 프레임 행 수 : {len(df)}")
df.to_excel(f"{name}_result.xlsx", index=False)