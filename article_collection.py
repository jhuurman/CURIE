#!/usr/bin/env python3
import os
import time
import json
import random
from urllib.parse import urljoin
import requests

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from bs4 import BeautifulSoup

class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


BASE_URL = "https://curie.pnnl.gov/curie2-combined-search"
LOGIN_URL = "https://curie.pnnl.gov/user/login"
DOWNLOAD_FOLDER = ""
TAGGED_FOLDER = os.path.join(DOWNLOAD_FOLDER, "Tagged_Documents")
UNTAGGED_FOLDER = os.path.join(DOWNLOAD_FOLDER, "Untagged_Documents")
NO_TAGS_FOLDER = os.path.join(DOWNLOAD_FOLDER, "No_Tags_Documents")
FAILED_DOWNLOADS_FILE = os.path.join(DOWNLOAD_FOLDER, "failed_downloads.txt")
CHECKPOINT_FILE = "checkpoint.json"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/91.0.4472.124 Safari/537.36")
}
GECKODRIVER_PATH = "/snap/bin/geckodriver" 

# make folders
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
os.makedirs(TAGGED_FOLDER, exist_ok=True)
os.makedirs(UNTAGGED_FOLDER, exist_ok=True)
os.makedirs(NO_TAGS_FOLDER, exist_ok=True)

def initialize_failed_downloads():
    with open(FAILED_DOWNLOADS_FILE, "w") as f:
        f.write("Failed Downloads:\n")
    print(f"{Colors.WARNING}[*] Failed downloads log initialized: {FAILED_DOWNLOADS_FILE}{Colors.ENDC}")

def log_failed_download(article_url, reason):
    with open(FAILED_DOWNLOADS_FILE, "a") as f:
        f.write(f"Article URL: {article_url}\nReason: {reason}\n\n")
    print(f"{Colors.WARNING}[!] Failed download logged: {article_url}{Colors.ENDC}")

def save_checkpoint(page_number, processed_links):
    checkpoint = {"page_number": page_number, "processed_links": list(processed_links)}
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f)
    print(f"{Colors.OKGREEN}[✓] Checkpoint saved: Page {page_number}{Colors.ENDC}")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            checkpoint = json.load(f)
        print(f"{Colors.OKCYAN}[*] Resuming from checkpoint: Page {checkpoint['page_number']}{Colors.ENDC}")
        return checkpoint["page_number"], set(checkpoint["processed_links"])
    else:
        print(f"{Colors.WARNING}[*] No checkpoint found. Starting from the beginning.{Colors.ENDC}")
        return 1, set()

def configure_driver():
    options = webdriver.FirefoxOptions()
    options.headless = True
    options.add_argument("--width=1920")
    options.add_argument("--height=1080")
    return webdriver.Firefox(service=Service(GECKODRIVER_PATH), options=options)

def perform_login(driver, username="", password=""):
    print(f"{Colors.HEADER}[*] Logging in as {username}...{Colors.ENDC}")
    driver.get(LOGIN_URL)
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "edit-name")))
    driver.find_element(By.ID, "edit-name").send_keys(username)
    driver.find_element(By.ID, "edit-pass").send_keys(password)
    driver.find_element(By.ID, "edit-submit").click()
    WebDriverWait(driver, 20).until_not(EC.presence_of_element_located((By.ID, "edit-name")))
    print(f"{Colors.OKGREEN}[✓] Login successful.{Colors.ENDC}")

def get_requests_session(driver):
    session = requests.Session()
    for cookie in driver.get_cookies():
        session.cookies.set(cookie['name'], cookie['value'])
    session.headers.update(HEADERS)
    return session

def get_article_links(driver, start_page=0, max_pages=0):
    print(f"{Colors.HEADER}[*] Retrieving article links starting from page {start_page}...{Colors.ENDC}")
    article_links = []
    current_page = start_page
    pages_processed = 0

    if current_page > 1:
        driver.get(f"{BASE_URL}?page={current_page - 1}")
    
    while True:
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "h2.title.curie-pseudo-title a"))
            )
        except Exception as e:
            print(f"{Colors.FAIL}[!] Timeout on page {current_page}: {e}{Colors.ENDC}")
            break

        link_elements = driver.find_elements(By.CSS_SELECTOR, "h2.title.curie-pseudo-title a")
        links = [elem.get_attribute("href") for elem in link_elements if elem.get_attribute("href")]
        article_links.extend(links)
        print(f"{Colors.OKCYAN}[INFO] Page {current_page}: Retrieved {len(links)} links.{Colors.ENDC}")
        pages_processed += 1

        if max_pages and pages_processed >= max_pages:
            print(f"{Colors.OKCYAN}[✓] Reached page limit: {max_pages}. Stopping pagination.{Colors.ENDC}")
            break

        try:
            next_page = driver.find_element(By.CSS_SELECTOR, "li.page-item.pager__item.pager__item--next a")
            next_page_url = next_page.get_attribute("href")
            if not next_page_url:
                print(f"{Colors.OKCYAN}[✓] No more pages available. Stopping pagination.{Colors.ENDC}")
                break
            driver.get(next_page_url)
            time.sleep(1.0)  
            current_page += 1
        except Exception as e:
            print(f"{Colors.OKCYAN}[✓] Pagination ended: {e}{Colors.ENDC}")
            break

    print(f"{Colors.OKGREEN}[✓] Total article links retrieved: {len(article_links)}; Last page processed: {current_page}{Colors.ENDC}")
    return article_links, current_page

def scrape_selected_subject_matters(driver, edit_url):
    try:
        driver.get(edit_url)
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[data-drupal-selector^='edit-field-nmw-subject-matter']"))
            )
        except Exception:
            print(f"{Colors.WARNING}[!] No subject matter checkboxes found on: {edit_url}.{Colors.ENDC}")
            return "no_subject", ""
        
        soup = BeautifulSoup(driver.page_source, "html.parser")
        checkboxes = soup.find_all("input", {"type": "checkbox"})
        subject_checkboxes = [cb for cb in checkboxes if "edit-field-nmw-subject-matter" in cb.get("data-drupal-selector", "")]
        
        if not subject_checkboxes:
            print(f"{Colors.WARNING}[!] No subject matter checkboxes found on: {edit_url}{Colors.ENDC}")
            return "no_subject", ""
        
        checked_boxes = [cb for cb in subject_checkboxes if cb.has_attr("checked")]
        if not checked_boxes:
            print(f"{Colors.WARNING}[!] Subject matter field present, but no boxes are checked on: {edit_url}{Colors.ENDC}")
            return "untagged", ""
        
        selected_labels = []
        for cb in checked_boxes:
            label = soup.find("label", {"for": cb.get("id")})
            label_text = label.get_text(strip=True) if label else cb.get("value", "")
            selected_labels.append(label_text)
        
        tag_string = "; ".join(selected_labels)
        print(f"{Colors.OKBLUE}[=] Extracted Tags: {tag_string}{Colors.ENDC}")
        return "tagged", tag_string

    except Exception as e:
        print(f"{Colors.FAIL}[!] Error scraping subject matters at {edit_url}: {e}{Colors.ENDC}")
        return "no_subject", ""

def extract_node_id_from_html(soup):
    article_tag = soup.find("article", attrs={"data-history-node-id": True})
    return article_tag["data-history-node-id"].strip() if article_tag else ""

def process_article(article_url, driver, session):
    status = "no_subject"
    pdf_url, subj_str, node_id = None, "", ""
    try:
        print(f"{Colors.OKCYAN}[*] Processing article: {article_url}{Colors.ENDC}")
        resp = session.get(article_url)
        if resp.status_code != 200:
            print(f"{Colors.FAIL}[!] Failed to fetch article page: {article_url}{Colors.ENDC}")
            log_failed_download(article_url, "Failed to fetch article page")
            return status, pdf_url, subj_str, node_id

        soup = BeautifulSoup(resp.text, "html.parser")
        pdf_link_tag = soup.select_one("div.field--name-field-document-attach-documents a[type='application/pdf']")
        if not pdf_link_tag:
            pdf_link_tag = soup.find("a", href=True, string=lambda s: s and "pdf" in s.lower())
        pdf_url = urljoin(article_url, pdf_link_tag["href"]) if pdf_link_tag else None

        node_id = extract_node_id_from_html(soup)
        if node_id:
            edit_url = f"https://curie.pnnl.gov/node/{node_id}/edit"
            status, subj_str = scrape_selected_subject_matters(driver, edit_url)
        else:
            print(f"{Colors.WARNING}[!] No node_id found; skipping subject matter scrape.{Colors.ENDC}")

        if pdf_url:
            print(f"{Colors.OKGREEN}[✓] Found PDF link: {pdf_url}{Colors.ENDC}")
        else:
            print(f"{Colors.WARNING}[!] No PDF link found on page: {article_url}{Colors.ENDC}")

    except Exception as e:
        print(f"{Colors.FAIL}[!] Error processing article {article_url}: {e}{Colors.ENDC}")
        log_failed_download(article_url, f"Error during processing: {e}")

    return status, pdf_url, subj_str, node_id

def download_pdf(pdf_url, target_folder, node_id, subject_matter_string):
    if not pdf_url or not node_id:
        print(f"{Colors.FAIL}[!] Invalid PDF URL or node_id for {pdf_url}{Colors.ENDC}")
        return False

    pdf_filename = f"{node_id}.pdf"
    param_filename = f"{node_id}_parameters.txt"
    pdf_path = os.path.join(target_folder, pdf_filename)
    param_path = os.path.join(target_folder, param_filename)

    os.makedirs(target_folder, exist_ok=True)

    if subject_matter_string.strip():
        try:
            with open(param_path, "w", encoding="utf-8") as f:
                f.write(subject_matter_string)
            print(f"{Colors.OKGREEN}[+] Written subject matter to: {param_path}{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.FAIL}[!] Error writing parameters file: {e}{Colors.ENDC}")
            return False
    else:
        print(f"{Colors.WARNING}[!] No subject matter data to write for {param_filename}{Colors.ENDC}")

    if os.path.exists(pdf_path):
        print(f"{Colors.OKBLUE}[=] File already exists: {pdf_path}{Colors.ENDC}")
        return True

    try:
        print(f"{Colors.OKGREEN}[+] Downloading PDF: {pdf_url}{Colors.ENDC}")
        # Set a timeout to prevent hanging indefinitely
        response = requests.get(pdf_url, headers=HEADERS, stream=True, timeout=30)
        if response.status_code == 200:
            with open(pdf_path, "wb") as pdf_file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        pdf_file.write(chunk)
            print(f"{Colors.OKGREEN}[✓] Download complete: {pdf_path}{Colors.ENDC}\n")
            return True
        else:
            print(f"{Colors.FAIL}[!] Failed to download PDF: {pdf_url}{Colors.ENDC}\n")
            log_failed_download(pdf_url, "Failed to download PDF")
            return False
    except Exception as e:
        print(f"{Colors.FAIL}[!] Error downloading PDF: {pdf_url}, {e}{Colors.ENDC}\n")
        log_failed_download(pdf_url, f"Error during download: {e}")
        return False

def main(max_pages):
    initialize_failed_downloads()
    driver = configure_driver()
    start_page, processed_links = load_checkpoint()

    try:
        perform_login(driver, username="", password="")
        driver.get(BASE_URL)
        session = get_requests_session(driver)

        article_links, last_page = get_article_links(driver, start_page, max_pages)
        new_links = [link for link in article_links if link not in processed_links]
        print(f"\n{Colors.BOLD}[*] Processing {len(new_links)} new articles...{Colors.ENDC}")

        for article_url in new_links:
            status, pdf_url, subj_str, node_id = process_article(article_url, driver, session)
            if pdf_url and node_id:
                if status == "tagged":
                    folder = TAGGED_FOLDER
                elif status == "untagged":
                    folder = UNTAGGED_FOLDER
                elif status == "no_subject":
                    folder = NO_TAGS_FOLDER
                else:
                    folder = UNTAGGED_FOLDER

                if download_pdf(pdf_url, folder, node_id, subj_str):
                    processed_links.add(article_url)
                    save_checkpoint(last_page, processed_links)  
        save_checkpoint(last_page, processed_links)
    finally:
        driver.quit()

    print(f"\n{Colors.BOLD}Summary:{Colors.ENDC}")
    print(f"{Colors.OKGREEN}[✓] Successfully processed and downloaded: {len(processed_links)} articles{Colors.ENDC}")

if __name__ == "__main__":
    main(max_pages=0)
