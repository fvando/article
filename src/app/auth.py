import streamlit as st
from src.core.config import ADMIN_USER, ADMIN_PASS
from src.core.i18n import t

def check_password():
    """Returns `True` if the user had a correct password."""

    def password_entered():
        if st.session_state["username"] == ADMIN_USER and st.session_state["password"] == ADMIN_PASS:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False


    if "password_correct" not in st.session_state:
        # First run, show inputs
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            st.divider()
            st.markdown(f"### {t('auth_restricted')}")
            st.caption(t('auth_desc'))
            st.text_input(t('auth_user'), key="username")
            st.text_input(t('auth_pass'), type="password", key="password", on_change=password_entered)
        return False
    
    elif not st.session_state["password_correct"]:
        # Password check failed
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            st.divider()
            st.markdown(f"### {t('auth_restricted')}")
            st.text_input(t('auth_user'), key="username")
            st.text_input(t('auth_pass'), type="password", key="password", on_change=password_entered)

            st.error(t('auth_error'))
        return False
    
    else:
        # Password correct
        return True

def logout():
    st.session_state["password_correct"] = False
    st.rerun()
