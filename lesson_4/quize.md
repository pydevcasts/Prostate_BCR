

### سوالات:

1. **🔢 سوال درباره شکل آرایه:**

   * **سوال:** یک آرایه دو بعدی 3x3 با استفاده از `np.arange` ایجاد کنید و سپس شکل آن آرایه را با استفاده از `shape` چاپ کنید.
   * **پاسخ:**

     ```python
     arr = np.arange(9).reshape(3, 3)
     print(arr.shape)
     ```

2. **➕ سوال درباره عملگرهای ریاضی:**

   * **سوال:** دو آرایه یک بعدی با مقادیر تصادفی بسازید (هر کدام 5 عدد) و مجموع آنها را با استفاده از عملگر جمع (`+`) محاسبه کنید.
   * **پاسخ:**

     ```python
     a = np.random.randint(1, 10, 5)
     b = np.random.randint(1, 10, 5)
     sum_arr = a + b
     print(sum_arr)
     ```

3. **🎲 سوال درباره توزیع تصادفی:**

   * **سوال:** 10 عدد تصادفی با توزیع نرمال (Gaussian) با میانگین 0 و انحراف معیار 1 تولید کنید. آرایه تولید شده را چاپ کنید.
   * **پاسخ:**

     ```python
     random_numbers = np.random.normal(0, 1, 10)
     print(random_numbers)
     ```

4. **✂️ سوال درباره برش آرایه:**

   * **سوال:** یک آرایه دو بعدی 4x4 ایجاد کنید و عناصر سطر دوم و ستون سوم را با استفاده از برش (slicing) استخراج کنید.
   * **پاسخ:**

     ```python
     arr = np.arange(16).reshape(4, 4)
     element = arr[1, 2]  # عنصر سطر دوم و ستون سوم
     print(element)
     ```

5. **📏 سوال درباره تابع `np.dot`:**

   * **سوال:** دو ماتریس 2x2 تعریف کنید و حاصلضرب آنها را با استفاده از تابع `np.dot` محاسبه کنید.
   * **پاسخ:**

     ```python
     a = np.array([[1, 2], [3, 4]])
     b = np.array([[5, 6], [7, 8]])
     result = np.dot(a, b)
     print(result)
     ```

### سوالات چالشی:

6. **📊 سوال درباره تجزیه و تحلیل داده‌ها:**

   * **سوال:** یک آرایه یک بعدی از اعداد تصادفی (10 عدد) ایجاد کنید و میانگین و انحراف معیار آن را محاسبه کنید.
   * **پاسخ:**

     ```python
     data = np.random.rand(10)
     mean = np.mean(data)
     std_dev = np.std(data)
     print("Mean:", mean, "Standard Deviation:", std_dev)
     ```

7. **🔍 سوال درباره شرطی‌سازی:**

   * **سوال:** یک آرایه یک بعدی با 10 عدد تصادفی ایجاد کنید و تعداد عناصری که بزرگتر از 0.5 هستند را شمارش کنید.
   * **پاسخ:**

     ```python
     arr = np.random.rand(10)
     count = np.sum(arr > 0.5)
     print("Count of elements greater than 0.5:", count)
     ```

### سوالات سطح سخت:

8. **📈 سوال درباره تجزیه و تحلیل داده‌ها:**

   * **سوال:** یک آرایه دو بعدی 5x5 بسازید که شامل اعداد تصادفی باشد. سپس بزرگ‌ترین و کوچک‌ترین مقدار را در آرایه پیدا کنید و مختصات (ایندکس) آن‌ها را چاپ کنید.
   * **پاسخ:**

     ```python
     arr = np.random.rand(5, 5)
     max_index = np.unravel_index(np.argmax(arr), arr.shape)
     min_index = np.unravel_index(np.argmin(arr), arr.shape)
     print("Max value:", arr[max_index], "at index:", max_index)
     print("Min value:", arr[min_index], "at index:", min_index)
     ```

9. **🔢 سوال درباره برش‌های پیچیده:**

   * **سوال:** یک آرایه دو بعدی 6x6 بسازید و سطرهای فرد (1، 3، 5) و ستون‌های زوج (0، 2، 4) آن را استخراج کنید.
   * **پاسخ:**

     ```python
     arr = np.arange(36).reshape(6, 6)
     sliced_arr = arr[1::2, ::2]
     print(sliced_arr)
     ```

10. **📊 سوال درباره محاسبات منطقی:**

    * **سوال:** یک آرایه یک بعدی با 20 عدد تصادفی بین 0 و 100 ایجاد کنید. سپس درصد عناصری که بزرگتر از 50 هستند را محاسبه کنید.
    * **پاسخ:**

      ```python
      arr = np.random.randint(0, 101, 20)
      percentage = np.sum(arr > 50) / arr.size * 100
      print("Percentage of elements greater than 50:", percentage)
      ```

11. **🔄 سوال درباره تبدیل نوع داده:**

    * **سوال:** یک آرایه 10 عددی از اعداد صحیح ایجاد کنید. آن را به نوع داده float تبدیل کنید و مجموع اعداد را محاسبه کنید.
    * **پاسخ:**

      ```python
      arr = np.random.randint(1, 10, 10)
      float_arr = arr.astype(float)
      total_sum = np.sum(float_arr)
      print("Sum of float array:", total_sum)
      ```

12. **⚙️ سوال درباره تابع `np.apply_along_axis`:**

    * **سوال:** یک آرایه دو بعدی 4x4 بسازید و با استفاده از `np.apply_along_axis`، مجموع هر ستون را محاسبه کنید.
    * **پاسخ:**

      ```python
      arr = np.random.randint(1, 10, (4, 4))
      column_sums = np.apply_along_axis(np.sum, 0, arr)
      print("Column sums:", column_sums)
      ```

13. **🔄 سوال درباره محاسبات پیچیده:**

    * **سوال:** یک آرایه 3x3 بسازید و معکوس آن را محاسبه کنید. اگر آرایه معکوس‌پذیر نبود، یک پیام خطا چاپ کنید.
    * **پاسخ:**

      ```python
      arr = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
      try:
          inv_arr = np.linalg.inv(arr)
          print("Inverse of the array:\n", inv_arr)
      except np.linalg.LinAlgError:
          print("The array is not invertible.")
      ```

14. **🔍 سوال درباره تابع `np.where`:**

    * **سوال:** یک آرایه یک بعدی از اعداد تصادفی بین 0 و 100 ایجاد کنید. سپس با استفاده از `np.where`، یک آرایه جدید بسازید که در آن اعداد بزرگتر از 50 با "High" و اعداد کمتر یا برابر با 50 با "Low" جایگزین شوند.
    * **پاسخ:**

      ```python
      arr = np.random.randint(0, 101, 10)
      labels = np.where(arr > 50, "High", "Low")
      print("Original array:", arr)
      print("Labels array:", labels)
      ```

15. **📈 سوال درباره محاسبات تجمعی:**

    * **سوال:** یک آرایه یک بعدی از 10 عدد تصادفی بین 1 تا 100 ایجاد کنید و مجموع تجمعی (cumulative sum) آن را محاسبه کنید.
    * **پاسخ:**

      ```python
      arr = np.random.randint(1, 101, 10)
      cumulative_sum = np.cumsum(arr)
      print("Original array:", arr)
      print("Cumulative sum:", cumulative_sum)
      ```




