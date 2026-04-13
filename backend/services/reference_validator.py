import re


class ReferenceValidator:
    def __init__(self):
        self.rules = {
            "M": self._validate_common,
            "C": self._validate_common,
            "G": self._validate_common,
            "N": self._validate_common,
            "J": self._validate_common,
            "D": self._validate_common,
            "R": self._validate_common,
            "S": self._validate_common,
            "P": self._validate_common,
            "DB": self._validate_common,
            "CP": self._validate_common,
            "EB": self._validate_common,
            "A": self._validate_common,
            "CM": self._validate_common,
            "DS": self._validate_common,
            "Z": self._validate_test,
            "M/OL": self._validate_test,
            "C/OL": self._validate_test,
            "G/OL": self._validate_test,
            "N/OL": self._validate_test,
            "J/OL": self._validate_test,
            "D/OL": self._validate_test,
            "R/OL": self._validate_test,
            "S/OL": self._validate_test,
            "P/OL": self._validate_test,
            "DB/OL": self._validate_test,
            "CP/OL": self._validate_test,
            "EB/OL": self._validate_test,
            "A/OL": self._validate_test,
            "CM/OL": self._validate_test,
            "DS/OL": self._validate_test,
            "Z/OL": self._validate_test,
        }

    def _parse_entry(self, entry_text):
        """改进后的解析方法"""
        # 清理干扰内容
        entry_text = re.sub(
            r"\(in\s*Chinese\)[\s\S]*?\.", "", entry_text, flags=re.I
        )  # 移除英文翻译
        entry_text = re.sub(r"\n\s*（责任编辑：.*$", "", entry_text)  # 清理尾部说明

        parsed = {}

        # 匹配文献类型标识
        # print(entry_text)
        type_match = re.search(r"\[([A-Z]{1,2}(?:/[A-Z]{1,2})?)\]", entry_text)

        return type_match.group(1) if type_match else None

    def _validate_test(self, entry):
        # 存储提取结果的字典
        fields = {}

        # 当前剩余待处理的文本
        remaining = entry.strip()

        title_match = re.search(
            r"\.\s*(?P<title>.+?)\s*\[([A-Z]{1,2}(?:/[A-Z]{1,2})?)\]", remaining
        )
        if title_match:
            fields["题名"] = title_match.group("title")
            author = remaining[: title_match.start()]
            remaining = remaining[title_match.end() :].strip()
            fields["主要责任者"] = author
        else:
            title_match = re.search(
                r"(?P<title>.+?)\s*\[([A-Z]{1,2}(?:/[A-Z]{1,2})?)\]", remaining
            )
            if title_match:
                fields["题名"] = title_match.group("title")
                remaining = remaining[title_match.end() :].strip()

        # 模式3：提取文献类型（方括号内的字母）
        type_match = re.search(
            r"(?P<type>\[([A-Z]{1,2}(?:/[A-Z]{1,2})?)\])(\s*/\s*/\s*)?(\.\s*)?", entry
        )
        if type_match:
            fields["文献类型"] = type_match.group("type")
            remaining = entry[type_match.end() :].strip()

        # 模式4：提取出版者（到句点结束）
        publisher_match = re.search(
            r"^(?P<publisher>.*?)\.(?=\s*(19|20)\s*\d{2})s*", remaining
        )
        # print(remaining)
        if publisher_match:
            fields["出版者"] = publisher_match.group("publisher")
            remaining = remaining[publisher_match.end() :].strip()
        else:
            publisher_match = re.search(
                r"^(?P<publisher>.*?),(?=\s*(19|20)\s*\d{2})s*", remaining
            )
            if publisher_match:
                fields["出版者"] = publisher_match.group("publisher")
                remaining = remaining[publisher_match.end() :].strip()
            else:
                publisher_match = re.search(
                    r"^(?P<publisher>.+?)(?=\.\s*(\d+\s*-\s*\d+))s*", remaining
                )
                if publisher_match:
                    fields["出版者"] = publisher_match.group("publisher")
                    remaining = remaining[publisher_match.end() :].strip()
                else:
                    publisher_match = re.search(
                        r"^(?P<publisher>.+?)(:|,)s*", remaining
                    )
                    if publisher_match:
                        fields["出版者"] = publisher_match.group("publisher")
                        remaining = remaining[publisher_match.end() :].strip()

        # print(remaining)
        # 模式5：提取年份（4位数字）
        year_match = re.search(r"(?P<year>(19|20)\s*\d{2})(\.|:|,)?\s*", remaining)
        if year_match:
            fields["出版年"] = year_match.group("year")
            remaining = remaining[year_match.end() :].strip()
        else:
            year_match = re.search(r"(?P<year>(19|20)\s*\d{2})(\.|:|,)?\s*", entry)
            if year_match:
                fields["出版年"] = year_match.group("year")
                remaining = entry[year_match.end() :].strip()

        # 模式6：提取章节号（数字(数字)）
        # print(remaining)
        chapter_match = re.search(r"(?P<chap>\d+\(\d+\))", remaining)
        if chapter_match:
            fields["章节号"] = chapter_match.group("chap")

        # 模式7：提取页码（数字-数字）
        pages_match = re.search(r"(?P<page>\d+\s*-\s*\d+)", remaining)
        if pages_match:
            fields["页码"] = pages_match.group("page")

        url_match = re.search(r"(?P<url>h\s*t\s*t\s*p\s*s?\s*://[^\s,]+)", entry)
        if url_match:
            fields["网址"] = url_match.group("url")

        arXiv_match = re.search(
            r"(?P<arXiv>arXiv:\s*(\d(?:\s*\d)*\.\s*\d(?:\s*\d)*)(\[[a-z]+\.[A-Z]+\])?.+?)",
            entry,
        )
        if arXiv_match:
            fields["arXiv"] = arXiv_match.group("arXiv")

        errors = []

        keys = fields.keys()
        if "出版者" in keys:
            fields["来源"] = fields["出版者"]
        elif "网址" in keys:
            fields["来源"] = fields["网址"]
        elif "arXiv" in keys:
            fields["来源"] = fields["arXiv"]

        # if True:#('http' in entry) and not '网址' in keys:
        # print(keys)
        # print(entry)
        # for field, msg in fields.items():
        # print(field,msg)

        required_fields = {
            "主要责任者": "缺失报告责任单位/作者",
            "题名": "缺失报告标题",
            "来源": "缺失来源（如：出版者，网址，arXiv）",
            "出版年": "缺失出版年份",
        }

        for field, msg in required_fields.items():
            if field not in keys:
                errors.append(msg)

        # if len(errors) > 0:
        # print(keys)
        # print(entry)
        # for field, msg in fields.items():
        # print(field,msg)
        # print(errors)
        return errors

    def _validate_common(self, entry):
        # 存储提取结果的字典
        fields = {}

        # 当前剩余待处理的文本
        remaining = entry.strip()

        title_match = re.search(
            r"\.\s*(?P<title>.+?)\s*\[([A-Z]{1,2}(?:/[A-Z]{1,2})?)\]", remaining
        )
        if title_match:
            fields["题名"] = title_match.group("title")
            author = remaining[: title_match.start()]
            remaining = remaining[title_match.end() :].strip()
            fields["主要责任者"] = author
        else:
            title_match = re.search(
                r"(?P<title>.+?)\s*\[([A-Z]{1,2}(?:/[A-Z]{1,2})?)\]", remaining
            )
            if title_match:
                fields["题名"] = title_match.group("title")
                remaining = remaining[title_match.end() :].strip()

        # 模式3：提取文献类型（方括号内的字母）
        type_match = re.search(
            r"(?P<type>\[([A-Z]{1,2}(?:/[A-Z]{1,2})?)\])(\s*/\s*/\s*)?(\.\s*)?", entry
        )
        if type_match:
            fields["文献类型"] = type_match.group("type")
            remaining = entry[type_match.end() :].strip()

        # 模式4：提取出版者（到句点结束）
        publisher_match = re.search(
            r"^(?P<publisher>.*?)\.(?=\s*(19|20)\s*\d{2})s*", remaining
        )
        # print(remaining)
        if publisher_match:
            fields["出版者"] = publisher_match.group("publisher")
            remaining = remaining[publisher_match.end() :].strip()
        else:
            publisher_match = re.search(
                r"^(?P<publisher>.*?),(?=\s*(19|20)\s*\d{2})s*", remaining
            )
            if publisher_match:
                fields["出版者"] = publisher_match.group("publisher")
                remaining = remaining[publisher_match.end() :].strip()
            else:
                publisher_match = re.search(
                    r"^(?P<publisher>.+?)(?=(\.|:|\[)\s*(\d+\s*-\s*\d+))s*", remaining
                )
                if publisher_match:
                    fields["出版者"] = publisher_match.group("publisher")
                    remaining = remaining[publisher_match.end() :].strip()

        # print(remaining)
        # 模式5：提取年份（4位数字）
        year_match = re.search(r"(?P<year>(19|20)\s*\d{2})(\.|:|,)?\s*", remaining)
        if year_match:
            fields["出版年"] = year_match.group("year")
            remaining = remaining[year_match.end() :].strip()
        else:
            year_match = re.search(r"(?P<year>(19|20)\s*\d{2})(\.|:|,)?\s*", entry)
            if year_match:
                fields["出版年"] = year_match.group("year")
                remaining = entry[year_match.end() :].strip()

        # 模式6：提取章节号（数字(数字)）
        # print(remaining)
        chapter_match = re.search(r"(?P<chap>\d+\(\d+\))", remaining)
        if chapter_match:
            fields["章节号"] = chapter_match.group("chap")

        # 模式7：提取页码（数字-数字）
        pages_match = re.search(r"(?P<page>\d+\s*-\s*\d+)", remaining)
        if pages_match:
            fields["页码"] = pages_match.group("page")

        url_match = re.search(r"(?P<url>h\s*t\s*t\s*p\s*s?\s*://.+?)(,|\s*)", entry)
        if url_match:
            fields["网址"] = url_match.group("url")

        arXiv_match = re.search(
            r"(?P<arXiv>arXiv:\s*(\d(?:\s*\d)*\.\s*\d(?:\s*\d)*)(\[[a-z]+\.[A-Z]+\])?.+?)",
            entry,
        )
        if arXiv_match:
            fields["arXiv"] = arXiv_match.group("arXiv")

        errors = []

        keys = fields.keys()

        if ("http" in entry) and not "网址" in keys:
            print(keys)
            print(entry)
            for field, msg in fields.items():
                print(field, msg)

        required_fields = {
            "主要责任者": "缺失报告责任单位/作者",
            "题名": "缺失报告标题",
            "出版者": "缺失出版机构",
            "出版年": "缺失出版年份",
        }

        for field, msg in required_fields.items():
            if field not in keys:
                errors.append(msg)

        return errors

    def _validate_unknown_online(self, entry):
        errors = []
        return errors

    def validate_reference(self, text):
        # 使用正则分割参考文献条目（示例匹配 [1]...格式）
        text = text.replace("\n", "")
        entries = re.split(r"\[\d+\]\s*", text)
        entries = [e for e in entries if e]
        num_entries = len(entries)
        # print(entries)
        validation_results = []

        for entry in entries:  # 跳过第一个空字符串
            # print(entry)
            type = self._parse_entry(entry)
            if not type:
                # validation_results.append({"entry": entry, "errors": ["格式解析失败"]})
                continue
            # if not "Z" in type:
            # continue
            # print(type)
            validator = self.rules.get(type)
            if validator:
                errors = validator(entry)
            else:
                # validation_results.append({"entry": entry, "errors": [f"未知格式{type}"]})
                continue
            if len(errors) > 0:
                validation_results.append({"entry": entry, "errors": errors})

        return validation_results, num_entries
