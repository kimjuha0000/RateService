import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

def crawl_all_movie_titles():
    """
    네이버에서 '현재상영영화' 검색 결과 페이지를 열고,
    여러 페이지를 순회하며 모든 영화 제목을 크롤링하는 함수
    """
    # 검색 결과 URL (필요에 따라 변경)
    url = "https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query=%ED%98%84%EC%9E%AC%EC%83%81%EC%98%81%EC%98%81%ED%99%94"
    
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # 브라우저 창 없이 실행하려면 주석 해제
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    time.sleep(2)  # 초기 페이지 로딩 대기

    all_titles = []

    while True:
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        
        # 영화 제목 추출 (예시 CSS 선택자; 실제 페이지 구조에 맞게 수정)
        # 제시하신 선택자에서 "div:nth-child(1)"을 제거하여 모든 항목 선택
        selector = (
            "div.cm_content_wrap > div > div > div > div.card_content._result_area "
            "> div.card_area._panel > div > div.data_area > div > div.title.multi_line._ellipsis > div > a"
        )
        title_tags = soup.select(selector)
        print("현재 페이지 영화 제목 개수:", len(title_tags))
        
        for tag in title_tags:
            title = tag.get_text(strip=True)
            if title:
                all_titles.append(title)
        
        # '다음' 버튼 클릭 (선택자 수정 필요)
        try:
            next_btn = driver.find_element(By.CSS_SELECTOR, "a.pg_next.on._next")
            next_btn.click()
            time.sleep(2)  # 페이지 전환 대기
        except Exception as e:
            print("더 이상 다음 페이지가 없거나, '다음' 버튼을 찾을 수 없습니다.", e)
            break

    driver.quit()
    return all_titles

def main():
    titles = crawl_all_movie_titles()
    print("총 크롤링한 영화 제목 개수:", len(titles))
    for idx, title in enumerate(titles, start=1):
        print(f"{idx}. {title}")

if __name__ == "__main__":
    main()
