// src/services/translationService.js
/**
 * Translation service for the robotics textbook
 * Provides basic Urdu translations for key elements
 */

class TranslationService {
  constructor() {
    this.translations = {
      ur: {
        // Navigation
        'Textbook': ' textbook',
        'Dashboard': 'ڈیش بورڈ',
        'GitHub': 'گِٹ ہب',
        
        // Common terms
        'Home': 'ہوم',
        'Module': 'ماڈیول',
        'Chapter': 'چیپٹر',
        'Learning Outcomes': 'سیکھنے کے نتائج',
        'Theory & Intuition': 'نظریہ اور جھلک',
        'Core Concepts': 'اہم تصورات',
        'Hands-On Labs': 'ہاتھوں ہاتھ کے معمے',
        'Sim-to-Real Notes': 'سیم ٹو ریئل نوٹس',
        'Multiple Choice Questions': 'کثیر انتخاب سوالات',
        'Further Reading': 'مزید پڑھائی',
        
        // Progress and completion
        'Completed': 'مکمل',
        'Mark Complete': 'مکمل نشان زد کریں',
        'Progress': 'پیشرفت',
        'Chapters Completed': 'مکمل کردہ ابواب',
        'Overall Progress': 'مجموعی پیشرفت',
        
        // Survey
        'Background Survey': 'پس منظر سروے',
        'Primary Goal': 'اہم مقصد',
        'Learning Pace': 'سیکھنے کی رفتار',
        
        // Recommendations
        'Next Recommended Chapter': 'اگلا تجویز کردہ باب',
        'Recently Completed': 'حال ہی میں مکمل',
        'Continue Learning': 'سیکھنا جاری رکھیں',
        'Chapters in Progress': 'زیر ترقی ابواب',
        
        // Chatbot
        'AI Assistant': 'ذہین اسسٹنٹ',
        'Ask about robotics concepts...': 'روبوٹکس کے تصورات کے بارے میں پوچھیں...',
        
        // Personalization
        'Personalization Dashboard': 'تخصیص کا ڈیش بورڈ',
        'Learning Preferences': 'سیکھنے کی ترجیحات',
        'Display Preferences': 'ڈسپلے کی ترجیحات',
        'Language Settings': 'زبان کی ترتیبات',
        'Font size': 'فونٹ کا سائز',
        'Slow': 'سست',
        'Moderate': 'اعتدال پسند',
        'Fast': 'تیز',
        'Small': 'چھوٹا',
        'Medium': 'متوسط',
        'Large': 'بڑا',
        'Enable dark mode': 'ڈارک موڈ فعال کریں',
        'Enable learning reminders': 'سیکھنے کی یاد دہانیاں فعال کریں',
        'Interface language': 'انٹرفیس کی زبان',
        'English': 'انگریزی',
        'Urdu': 'اردو',
        
        // Survey options
        'Beginner - Just starting out': 'شروع کار - ابھی ابتداء',
        'Intermediate - Some experience': 'درمیانہ - کچھ تجربہ',
        'Advanced - Significant experience': 'اعلیٰ - قابل ذکر تجربہ',
        'Expert - Professional/research level': 'ماہر - پیشہ ورانہ / تحقیقی سطح',
        'Student': 'طلبہ',
        'Robotics Engineer': 'روبوٹکس انجینئر',
        'Researcher': 'محقق',
        'Software Developer': 'سافٹ ویئر ڈیولپر',
        'Hobbyist/Maker': ' hobbyist/سازندہ',
        'Other': 'دیگر',
        'Slow - I prefer detailed explanations and take my time': 'سست - مجھے تفصیلی وضاحتیں پسند ہیں اور میں اپنا وقت لیتا ہوں',
        'Moderate - Good pace, balanced detail': 'اعتدال پسند - اچھی رفتار، متوازن تفصیل',
        'Fast - I prefer to move quickly through material': 'تیز - مجھے مواد کے ذریعے جلدی سے بڑھنا پسند ہے',
        'Less than 2 hours': '2 گھنٹے سے کم',
        '2-5 hours': '2-5 گھنٹے',
        '5-10 hours': '5-10 گھنٹے',
        'More than 10 hours': '10 گھنٹوں سے زیادہ',
        'None or very basic': 'کوئی نہیں یا بہت بنیادی',
        'Basic - Can write simple programs': 'بنیادی - سادہ پروگرام لکھ سکتا ہے',
        'Intermediate - Comfortable with programming concepts': 'درمیانہ - پروگرامنگ کے تصورات کے ساتھ مطابقت',
        'Advanced - Experienced with multiple languages': 'اعلیٰ - متعدد زبانوں کے ساتھ تجربہ کار',
        'None - First time working with robotics': 'کوئی نہیں - روبوٹکس کے ساتھ کام کرتے ہوئے پہلی بار',
        'Simulation - Only worked in simulation': 'سی뮬یشن - صرف سی뮬یشن میں کام کیا',
        'Basic hardware - Simple robots': 'بنیادی ہارڈ ویئر - سادہ روبوٹس',
        'Advanced - Complex robot systems': 'اعلیٰ - پیچیدہ روبوٹ سسٹمز',
        'Algebra/Trigonometry level': 'الجبر / مثلثات کی سطح',
        'Calculus level': 'حسابان کی سطح',
        'Linear algebra level': 'لکیری الجبر کی سطح',
        'Advanced - Graduate level math': 'اعلیٰ - گریجویٹ سطح کی ریاضی',
        'Career advancement in robotics': 'روبوٹکس میں کیریئر کی ترقی',
        'Academic education or research': 'علمی تعلیم یا تحقیق',
        'Personal or hobby projects': 'ذاتی یا hobby منصوبے',
        'Research in AI/Robotics': 'AI/Robotics میں تحقیق',
        'Starting a robotics company': 'روبوٹکس کمپنی شروع کرنا',
        
        // General
        'Question': 'سوال',
        'of': 'کا',
        'Previous': 'گزشتہ',
        'Next': 'اگلے',
        'Complete Survey': 'سروے مکمل کریں',
        'Thank you for completing the survey! Your learning experience will be personalized based on your responses.': 'سروے مکمل کرنے کا شکریہ! آپ کا سیکھنے کا تجربہ آپ کے جوابات کی بنیاد پر ترتیب دیا جائے گا۔',
        'Survey Completed': 'سروے مکمل ہو گیا',
        'Survey Incomplete': 'سروے نامکمل',
        'Complete the background survey to personalize your learning experience and receive recommendations tailored to your needs.': 'اپنے سیکھنے کے تجربے کو ترتیب دینے اور اپنی ضروریات کے مطابق سفارشات حاصل کرنے کے لیے پس منظر کا سروے مکمل کریں۔',
        'Keep up the great work! Your learning pace is well-suited to your goals.': 'بہترین کام جاری رکھیں! آپ کی سیکھنے کی رفتار آپ کے اہداف کے لیے اچھی ہے۔',
        
        // Chapter navigation
        'Chapter Navigation': 'چیپٹر نیویگیشن',
        'Chapter Navigation': 'چیپٹر نیویگیشن',
      }
    };
  }

  /**
   * Get translation for a given key in the specified language
   * @param {string} key - The text to translate
   * @param {string} lang - The target language code (e.g., 'ur')
   * @returns {string} - The translated text or original if not found
   */
  getTranslation(key, lang = 'ur') {
    if (!this.translations[lang]) {
      return key; // Return original if language not supported
    }
    
    // Look for exact translation
    if (this.translations[lang][key]) {
      return this.translations[lang][key];
    }
    
    // If not found, return the original key
    return key;
  }

  /**
   * Translate content based on current language
   * @param {string} content - The content to translate
   * @returns {string} - Translated or original content
   */
  translateContent(content, lang = 'ur') {
    if (lang === 'en') return content; // If English, return as is
    
    // For now, we'll return translated version of common terms
    // In a real implementation, this would perform more comprehensive translation
    let translated = content;
    
    for (const [original, translatedText] of Object.entries(this.translations[lang])) {
      translated = translated.split(original).join(translatedText);
    }
    
    return translated;
  }
  
  /**
   * Check if a language is supported
   * @param {string} lang - Language code to check
   * @returns {boolean} - Whether language is supported
   */
  isLanguageSupported(lang) {
    return !!this.translations[lang];
  }
}

// Create a singleton instance
const translationService = new TranslationService();
window.TranslationService = translationService;

export default translationService;