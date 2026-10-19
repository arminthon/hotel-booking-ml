import streamlit as st
import pandas as pd
import joblib

# تنظیمات اولیه ظاهر صفحه
st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="centered"
)

# عنوان و توضیحات بالای صفحه
st.title("🏨 پیش‌بینی کنسلی رزرو هتل")
st.write("مشخصات رزرو مسافر را وارد کنید تا احتمال لغو شدن رزرو محاسبه شود.")
st.markdown("---")

# تابع کش‌شده برای خواندن مدل (تا با هر بار کلیک، مدل دوباره از هارد لود نشود)
@st.cache_resource
def load_saved_model():
    return joblib.load("hotel_cancel_model.pkl")

model = load_saved_model()

# فرم دریافت مشخصات مهم مسافر
with st.form("hotel_form"):
    st.subheader("📋 اطلاعات رزرو")
    col1, col2 = st.columns(2)

    with col1:
        hotel_type = st.selectbox("نوع هتل", ["City Hotel", "Resort Hotel"])
        lead_time = st.number_input("روزهای فاصله تا ورود (Lead Time)", min_value=0, max_value=800, value=60)
        stays_in_week_nights = st.number_input("تعداد شب‌های اقامت در روزهای هفته", min_value=0, max_value=50, value=3)
        stays_in_weekend_nights = st.number_input("تعداد شب‌های اقامت در آخر هفته", min_value=0, max_value=20, value=1)
        adults = st.number_input("تعداد بزرگسالان", min_value=1, max_value=20, value=2)
        children = st.number_input("تعداد کودکان", min_value=0, max_value=10, value=0)
        previous_cancellations = st.number_input("تعداد لغو‌های قبلی مسافر", min_value=0, max_value=30, value=0)

    with col2:
        deposit_type = st.selectbox("نوع پیش‌پرداخت", ["No Deposit", "Non Refund", "Refundable"])
        customer_type = st.selectbox("نوع مشتری", ["Transient", "Contract", "Transient-Party", "Group"])
        adr = st.number_input("میانگین قیمت هر شب (€)", min_value=0.0, max_value=5000.0, value=105.0)
        booking_changes = st.number_input("تعداد دفعات تغییر رزرو", min_value=0, max_value=20, value=0)
        required_car_parking_spaces = st.number_input("تعداد جای پارک درخواستی", min_value=0, max_value=10, value=0)
        total_of_special_requests = st.number_input("تعداد درخواست‌های ویژه", min_value=0, max_value=10, value=1)
        is_repeated_guest = st.selectbox(
            "آیا مسافر قبلی هتل است؟",
            [0, 1],
            format_func=lambda x: "بله" if x == 1 else "خیر"
        )

    submitted = st.form_submit_button("🔮 بررسی وضعیت رزرو")

# پردازش و پیش‌بینی پس از زدن دکمه
if submitted:
    # ۱. خواندن لیست ستون‌هایی که مدل آموزش دیده انتظار دارد
    expected_columns = getattr(model, "feature_names_in_", None)

    # ۲. ساخت یک ردیف با مقدار اولیه ۰.۰ (float) برای تمام ستون‌های مدل
    #    چرا float؟ چون بعضی ویژگی‌ها مثل adr اعشاری هستند و Pandas جدید اجازه نمی‌دهد
    #    روی ستون int مقدار float ست شود.
    if expected_columns is not None:
        input_df = pd.DataFrame(0.0, index=[0], columns=expected_columns)
    else:
        st.error("ستون‌های مدل شناسایی نشد!")
        st.stop()

    # ۳. پر کردن ستون‌های عددی مستقیم (همه را به float تبدیل می‌کنیم تا نوع داده یکدست باشد)
    direct_numeric_cols = {
        "lead_time": lead_time,
        "stays_in_week_nights": stays_in_week_nights,
        "stays_in_weekend_nights": stays_in_weekend_nights,
        "adults": adults,
        "children": children,
        "previous_cancellations": previous_cancellations,
        "adr": adr,
        "booking_changes": booking_changes,
        "required_car_parking_spaces": required_car_parking_spaces,
        "total_of_special_requests": total_of_special_requests,
        "is_repeated_guest": is_repeated_guest
    }

    for col_name, value in direct_numeric_cols.items():
        if col_name in input_df.columns:
            # با loc ست می‌کنیم و مقدار را float می‌کنیم تا Pandas گیر نوع داده ندهد
            input_df.loc[0, col_name] = float(value)

    # ۴. ست کردن ستون‌های One-Hot بر اساس انتخاب کاربر (اینجا هم 1.0 می‌گذاریم)
    dummy_choices = [
        f"hotel_{hotel_type}",
        f"deposit_type_{deposit_type}",
        f"customer_type_{customer_type}"
    ]

    for col_name in dummy_choices:
        if col_name in input_df.columns:
            input_df.loc[0, col_name] = 1.0

    # (اختیاری ولی مفید) اطمینان از اینکه کل دیتافریم float است
    input_df = input_df.astype(float)

    # ۵. انجام پیش‌بینی با مدل هوش مصنوعی
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    cancel_prob = probabilities[1] * 100
    keep_prob = probabilities[0] * 100

    st.markdown("---")
    # ۶. نمایش خروجی به کاربر
    if prediction == 1:
        st.error(f"⚠️ **این رزرو احتمالاً لغو خواهد شد!** (احتمال لغو: {cancel_prob:.1f}%)")
    else:
        st.success(f"✅ **این رزرو تأیید است و مهمان خواهد آمد.** (احتمال حضور: {keep_prob:.1f}%)")
