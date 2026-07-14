PBHI EMPLOYEE ATTRITION DASHBOARD

Cara menjalankan dashboard:

1. Extract seluruh isi ZIP ke satu folder.
2. Pastikan Python sudah terpasang.
3. Buka folder project.
4. Klik kanan area kosong di folder, lalu pilih:
   Open in Terminal

5. Buat virtual environment:

python -m venv .venv

6. Install library yang dibutuhkan:

.\.venv\Scripts\python.exe -m pip install -r requirements.txt

7. Jalankan dashboard:

.\.venv\Scripts\python.exe -m streamlit run app.py

8. Dashboard akan terbuka otomatis di browser.

Catatan:
- Gunakan file attrition_upload_template.csv sebagai contoh format upload.
- Jangan mengubah nama file model atau metadata.
- Pastikan seluruh file tetap berada dalam folder yang sama.
- Dashboard ini menggunakan data demo dan ditujukan sebagai decision-support tool.