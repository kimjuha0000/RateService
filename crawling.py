import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

import urllib.parse

def encode_movie_name(movie_name: str) -> str:
    return urllib.parse.quote(movie_name)






def scroll_reviews(driver):
    """
    'div.lego_review_list._scroller' 컨테이너를 반복 스크롤하여
    추가 리뷰가 로딩될 때까지 대기합니다.
    """
    container = driver.find_element(By.CSS_SELECTOR, "div.lego_review_list._scroller")
    last_height = driver.execute_script("return arguments[0].scrollHeight", container)
    
    while True:
        # 컨테이너의 최하단으로 스크롤
        driver.execute_script("arguments[0].scrollTo(0, arguments[0].scrollHeight);", container)
        time.sleep(3)  # 로딩 대기 (필요 시 조정)
        
        new_height = driver.execute_script("return arguments[0].scrollHeight", container)
        if new_height == last_height:
            # 더 이상 늘어나지 않으면 스크롤 종료
            break
        last_height = new_height


def test_crawling(encoded):
    url = "https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&ssc=tab.nx.all&query="+encoded  # 실제 URL로 변경
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    time.sleep(2)  # 페이지 로딩 대기

    try:
        latest_btn = driver.find_element(By.LINK_TEXT, "최신순")
        latest_btn.click()
        time.sleep(2)  # 최신순 정렬 로딩 대기
    except Exception as e:
        print("최신순 버튼 클릭 실패:", e)

    scroll_reviews(driver)

    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    review_items = soup.select(
        "div.cm_content_wrap > div > div > div:nth-child(4) > "
        "div.lego_review_list._scroller > ul > li"
    )
    print("전체 리뷰 개수:", len(review_items))
    
    reviews = []
    for li in review_items:
        review_text_elem = li.select_one("span.desc._text")
        review_text = review_text_elem.get_text(strip=True) if review_text_elem else ""
        time_elem = li.select_one("dd.this_text_normal")
        review_time = time_elem.get_text(strip=True) if time_elem else ""
        reviews.append({
            "review_text": review_text,
            "review_time": review_time
        })
    
    driver.quit()
    # 디버깅: 리뷰 데이터 확인
    print("크롤링된 리뷰 리스트:", reviews)
    return reviews





if __name__ == '__main__':
    movie = input("영화 제목을 입력하세요: ")
    encoded = encode_movie_name(movie+"관람평")
    test_crawling()
    print(movie)